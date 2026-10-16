package adapterbuild

import (
	"context"
	"fmt"
	"strings"

	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
)

// MinioStore is the ObjectStore over any S3-compatible service (MinIO locally).
type MinioStore struct {
	cfg    StoreConfig
	client *minio.Client
}

// NewMinioStore connects and checks the bucket exists. It never creates the
// bucket: publishing into a bucket nobody provisioned is a configuration
// mistake worth stopping on.
func NewMinioStore(ctx context.Context, cfg StoreConfig) (*MinioStore, error) {
	cl, err := minio.New(cfg.Endpoint, &minio.Options{
		Creds:  credentials.NewStaticV4(cfg.AccessKey, cfg.SecretKey, ""),
		Secure: cfg.Secure,
		Region: cfg.Region,
	})
	if err != nil {
		return nil, fmt.Errorf("adapterbuild: object store client: %w", err)
	}
	ok, err := cl.BucketExists(ctx, cfg.Bucket)
	if err != nil {
		return nil, fmt.Errorf("adapterbuild: checking bucket %q at %s: %w", cfg.Bucket, cfg.Endpoint, err)
	}
	if !ok {
		return nil, fmt.Errorf("adapterbuild: bucket %q does not exist at %s — create it first", cfg.Bucket, cfg.Endpoint)
	}
	return &MinioStore{cfg: cfg, client: cl}, nil
}

// Stat implements ObjectStore.
func (s *MinioStore) Stat(ctx context.Context, key string) (ObjectInfo, bool, error) {
	oi, err := s.client.StatObject(ctx, s.cfg.Bucket, key, minio.StatObjectOptions{})
	if err != nil {
		if r := minio.ToErrorResponse(err); r.Code == "NoSuchKey" || r.StatusCode == 404 {
			return ObjectInfo{}, false, nil
		}
		return ObjectInfo{}, false, err
	}
	meta := map[string]string{}
	for k, v := range oi.UserMetadata {
		meta[k] = v
	}
	// Some servers surface user metadata only as X-Amz-Meta-* headers.
	for k, vs := range oi.Metadata {
		if strings.HasPrefix(strings.ToLower(k), "x-amz-meta-") && len(vs) > 0 {
			meta[k] = vs[0]
		}
	}
	return ObjectInfo{Size: oi.Size, Metadata: meta, ETag: oi.ETag}, true, nil
}

// PutFile implements ObjectStore.
func (s *MinioStore) PutFile(ctx context.Context, key, path, contentType string, meta map[string]string) error {
	_, err := s.client.FPutObject(ctx, s.cfg.Bucket, key, path, minio.PutObjectOptions{
		ContentType:  contentType,
		UserMetadata: meta,
	})
	return err
}

// URL implements ObjectStore.
func (s *MinioStore) URL(key string) string { return s.cfg.PublicBaseURL + "/" + key }

// Describe implements ObjectStore.
func (s *MinioStore) Describe() string { return s.cfg.Bucket + "@" + s.cfg.Endpoint }
