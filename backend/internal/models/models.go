package models

import (
	"time"

	"gorm.io/gorm"
)

type User struct {
	ID        uint           `json:"id" gorm:"primaryKey"`
	Email     string         `json:"email" gorm:"uniqueIndex;not null"`
	Password  string         `json:"-" gorm:"not null"`
	Name      string         `json:"name"`
	CreatedAt time.Time      `json:"created_at"`
	UpdatedAt time.Time      `json:"updated_at"`
	DeletedAt gorm.DeletedAt `json:"-" gorm:"index"`
}

type Equation struct {
	ID         uint           `json:"id" gorm:"primaryKey"`
	UserID     uint           `json:"user_id" gorm:"not null"`
	User       User           `json:"user" gorm:"foreignKey:UserID"`
	Input      string         `json:"input" gorm:"not null"`
	Normalized string         `json:"normalized" gorm:"not null"`
	ImageData  string         `json:"image_data" gorm:"type:text"`
	Confidence float64        `json:"confidence"`
	CreatedAt  time.Time      `json:"created_at"`
	UpdatedAt  time.Time      `json:"updated_at"`
	DeletedAt  gorm.DeletedAt `json:"-" gorm:"index"`
}

type Solution struct {
	ID         uint           `json:"id" gorm:"primaryKey"`
	EquationID uint           `json:"equation_id" gorm:"not null"`
	Equation   Equation       `json:"equation" gorm:"foreignKey:EquationID"`
	Steps      string         `json:"steps" gorm:"type:text"`    // JSON array as string
	Solution   string         `json:"solution" gorm:"type:text"` // JSON array as string
	CreatedAt  time.Time      `json:"created_at"`
	UpdatedAt  time.Time      `json:"updated_at"`
	DeletedAt  gorm.DeletedAt `json:"-" gorm:"index"`
}

// Request/Response DTOs
type SolveEquationRequest struct {
	EquationText string `json:"equation_text" binding:"required"`
}

type SolveEquationResponse struct {
	Input      string   `json:"input"`
	Normalized string   `json:"normalized"`
	Steps      []string `json:"steps"`
	Solution   []string `json:"solution"`
	Confidence float64  `json:"confidence"`
}

type LoginRequest struct {
	Email    string `json:"email" binding:"required,email"`
	Password string `json:"password" binding:"required"`
}

type RegisterRequest struct {
	Email    string `json:"email" binding:"required,email"`
	Password string `json:"password" binding:"required,min=6"`
	Name     string `json:"name" binding:"required"`
}

type AuthResponse struct {
	Token string `json:"token"`
	User  User   `json:"user"`
}

type HistoryItem struct {
	ID         uint      `json:"id"`
	Equation   string    `json:"equation"`
	Solution   []string  `json:"solution"`
	Timestamp  time.Time `json:"timestamp"`
	Confidence float64   `json:"confidence"`
}
