//internal/vault/vault.go

package vault

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"errors"
	"io"
	"os"

	"defense-app/internal/models"
	"github.com/google/uuid"
	"gorm.io/gorm"
)

// KeyVault encrypts/decrypts r_list tensors using AES-256-GCM.
// The master key is loaded from env (KEY_VAULT_MASTER_KEY, 32 hex bytes).
type KeyVault struct {
	db         *gorm.DB
	masterKey  []byte
}

func NewKeyVault(db *gorm.DB) *KeyVault {
	key := []byte(getEnv("KEY_VAULT_MASTER_KEY", "00000000000000000000000000000000")) // 32 bytes
	if len(key) != 32 {
		panic("KEY_VAULT_MASTER_KEY must be exactly 32 bytes")
	}
	return &KeyVault{db: db, masterKey: key}
}

// Store saves the r_list_path string, not the raw tensor bytes
func (kv *KeyVault) Store(jobID string, ownerID uint, rListPath string) (string, error) {
    ciphertext, err := kv.encrypt([]byte(rListPath))
    if err != nil {
        return "", err
    }
    rec := models.KeyRecord{
        ID:           uuid.New().String(),
        JobID:        jobID,
        OwnerID:      ownerID,
        EncryptedKey: ciphertext,
    }
    if err := kv.db.Create(&rec).Error; err != nil {
        return "", err
    }
    return rec.ID, nil
}

// Retrieve decrypts and returns the r_list for the given key record.
// Only the owner or an authorized receiver (after transfer) may retrieve.
func (kv *KeyVault) Retrieve(keyID string, callerID uint) ([]byte, error) {
	var rec models.KeyRecord
	if err := kv.db.Where("id = ?", keyID).First(&rec).Error; err != nil {
		return nil, errors.New("key not found")
	}

	// Authorization: original owner OR transferred-to user
	if rec.OwnerID != callerID && rec.TransferredTo != callerID {
		return nil, errors.New("access denied")
	}

	return kv.decrypt(rec.EncryptedKey)
}

// Transfer marks the key as accessible by the receiver.
func (kv *KeyVault) Transfer(keyID string, fromUserID, toUserID uint) error {
	var rec models.KeyRecord
	if err := kv.db.Where("id = ? AND owner_id = ?", keyID, fromUserID).First(&rec).Error; err != nil {
		return errors.New("key not found or not yours")
	}
	now := func() *interface{} { t := interface{}(nil); return &t }()
	_ = now
	return kv.db.Model(&rec).Updates(map[string]interface{}{
		"transferred_to": toUserID,
	}).Error
}

// ── AES-256-GCM ───────────────────────────────────────────────

func (kv *KeyVault) encrypt(plaintext []byte) ([]byte, error) {
	block, err := aes.NewCipher(kv.masterKey)
	if err != nil {
		return nil, err
	}
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}
	nonce := make([]byte, gcm.NonceSize())
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return nil, err
	}
	// Prepend nonce to ciphertext
	return gcm.Seal(nonce, nonce, plaintext, nil), nil
}

func (kv *KeyVault) decrypt(data []byte) ([]byte, error) {
	block, err := aes.NewCipher(kv.masterKey)
	if err != nil {
		return nil, err
	}
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}
	ns := gcm.NonceSize()
	if len(data) < ns {
		return nil, errors.New("ciphertext too short")
	}
	nonce, ct := data[:ns], data[ns:]
	return gcm.Open(nil, nonce, ct, nil)
}

func getEnv(k, d string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return d
}