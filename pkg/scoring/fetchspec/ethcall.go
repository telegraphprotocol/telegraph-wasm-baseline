package fetchspec

import (
	"context"
	"encoding/json"
	"fmt"
	"math/big"
	"net/http"
	"strconv"
	"strings"
)

// ethCall reads a value straight out of an EVM contract.
//
// This is the transport for a quantity that has no publisher. A lending rate
// is not reported by anyone — it IS the contract's state, so reading it is the
// only way to get a number that no miner in the round also produced. Every
// yield API we could reach is either one of the registered miners or a
// redistributor of one, which makes them oracles: a miner scored against its
// own upstream cannot lose.
//
// The spec supplies the call in Params:
//
//	to     — the contract address (0x…)
//	data   — the ABI-encoded call, selector plus arguments
//	word   — which 32-byte word of the return to read (default 0)
//	divisor— decimal string the word is divided by (default 1)
//	rpc    — the JSON-RPC endpoint; URL is used when this is absent
//
// word exists because a Solidity function returning a struct returns its
// fields concatenated, with no framing: Aave's getReserveData packs the
// liquidity rate at word 2 of ten. Reading word 0 of that response yields a
// configuration bitmap — a large, plausible, completely wrong number.
type ethCall struct{}

func init() { Register(EthCall, ethCall{}) }

func (ethCall) Fetch(ctx context.Context, spec Spec, params map[string]string) ([]byte, error) {
	// The call can be declared per-QUESTION as well as per-intent. A spec whose
	// questions are different pools reads a different contract for each, so
	// "to" and "data" belong with the question, not with the intent — and
	// SpecTruthSource passes a question's key_params in here. Question params
	// win; Spec.Params is the fallback for a spec where every question hits
	// the same contract.
	pick := func(name string) string {
		if v, ok := params[name]; ok && v != "" {
			return v
		}
		v, _ := spec.Params[name].(string)
		return v
	}
	to, data := pick("to"), pick("data")
	if to == "" || data == "" {
		return nil, fmt.Errorf("fetchspec: eth_call needs params.to and params.data")
	}
	endpoint := spec.URL
	if r := pick("rpc"); r != "" {
		endpoint = r
	}
	if endpoint == "" {
		return nil, fmt.Errorf("fetchspec: eth_call needs an rpc endpoint (url or params.rpc)")
	}

	payload := fmt.Sprintf(
		`{"jsonrpc":"2.0","id":1,"method":"eth_call","params":[{"to":%q,"data":%q},"latest"]}`,
		to, data)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, strings.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("fetchspec: building the eth_call request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "TelegraphNode/1.0")

	resp, err := clientFor(ctx).Do(req)
	if err != nil {
		return nil, fmt.Errorf("fetchspec: eth_call to %s: %w", redact(endpoint), err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("fetchspec: eth_call endpoint returned %s", resp.Status)
	}

	var out struct {
		Result string `json:"result"`
		Error  *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("fetchspec: decoding the eth_call response: %w", err)
	}
	if out.Error != nil {
		// A revert is not a zero. Returning one would score every miner
		// against a truth of 0 and rank the most wrong answer first.
		return nil, fmt.Errorf("fetchspec: eth_call reverted: %s", out.Error.Message)
	}

	hex := strings.TrimPrefix(out.Result, "0x")
	if hex == "" {
		return nil, fmt.Errorf("fetchspec: eth_call returned no data (wrong selector, or not a contract)")
	}

	word := 0
	if w := pick("word"); w != "" {
		n, err := strconv.Atoi(w)
		if err != nil {
			return nil, fmt.Errorf("fetchspec: word %q is not an integer", w)
		}
		word = n
	} else if w, ok := spec.Params["word"].(float64); ok {
		word = int(w)
	}
	lo, hi := word*64, (word+1)*64
	if hi > len(hex) {
		return nil, fmt.Errorf("fetchspec: eth_call returned %d words, spec asked for word %d",
			len(hex)/64, word)
	}

	v, ok := new(big.Int).SetString(hex[lo:hi], 16)
	if !ok {
		return nil, fmt.Errorf("fetchspec: word %d is not hex: %q", word, hex[lo:hi])
	}

	// The divisor converts a fixed-point on-chain integer to its real value
	// (Aave rates are rays, 1e27). Done as a rational so no precision is lost
	// before Extract applies the intent's own scale.
	out2 := new(big.Float).SetInt(v)
	if d := pick("divisor"); d != "" {
		div, ok := new(big.Float).SetString(d)
		if !ok || div.Sign() == 0 {
			return nil, fmt.Errorf("fetchspec: params.divisor %q is not a positive number", d)
		}
		out2.Quo(out2, div)
	}

	// Handed on as JSON so Extract's dotted-path reader works unchanged, the
	// same way the ZIP fetcher converts CSV before returning.
	f, _ := out2.Float64()
	return []byte(fmt.Sprintf(`{"value":%v}`, f)), nil
}
