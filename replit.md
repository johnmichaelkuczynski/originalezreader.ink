# Multi-Provider AI Text Processing Application

## Overview

This is a Flask-based web application that provides advanced text processing capabilities using multiple AI providers (OpenAI, Anthropic, Perplexity, DeepSeek). The application specializes in text rewriting, translation, style transfer, and document processing with support for various file formats including PDFs, Word documents, and audio files.

## System Architecture

### Backend Architecture
- **Framework**: Flask web framework with SQLAlchemy ORM
- **Database**: SQLite (can be easily switched to PostgreSQL)
- **AI Processing**: Multi-provider architecture with automatic failover
- **File Processing**: Support for PDF, DOCX, images (OCR), and audio transcription
- **Session Management**: Flask-Login for user authentication
- **API Integration**: Multiple AI providers with key rotation and rate limiting

### Frontend Architecture
- **Template Engine**: Jinja2 templates
- **UI Components**: Bootstrap-based responsive design
- **File Upload**: Drag-and-drop interface with multiple format support
- **Real-time Processing**: AJAX-based asynchronous processing with progress tracking

## Key Components

### 1. Multi-Provider AI Processing (`multi_provider_processor.py`)
- Handles text processing across multiple AI providers
- Implements two-level chunking (macrochunks and subchunks)
- Automatic failover between providers
- Rate limiting and error handling
- Support for different processing actions (rewrite, translate, summarize)

### 2. API Key Management (`api_key_manager.py`)
- Manages multiple API keys per provider
- Health tracking and automatic rotation
- Rate limit detection and recovery
- Load balancing across available keys

### 3. File Processing System
- **PDF Processing**: Text extraction using PyPDF2
- **Word Documents**: Content extraction using python-docx
- **Image Processing**: OCR using pytesseract
- **Audio Transcription**: Speech-to-text using SpeechRecognition and OpenAI Whisper

### 4. Database Models (`models.py`)
- **UserProfile**: Stores user information and writing samples
- **TextEntry**: Tracks processed documents and their metadata
- **WritingSample**: Stores user writing samples for style analysis
- **ContentSource**: Manages uploaded documents and their content

### 5. Translation System (`simple_translation.py`)
- Multi-provider translation with automatic chunking
- Language detection and validation
- Support for 9+ languages including English, Spanish, French, German, Italian, Portuguese, Dutch, Russian, and Chinese

### 6. Style Transfer System (`style_rewrite_passthrough.py`)
- Implements style rewriting using user-provided samples
- Pass-through architecture for maximum fidelity
- Supports academic, creative, and technical writing styles

## Data Flow

1. **Document Upload**: Users upload files through the web interface
2. **Content Extraction**: System extracts text from various file formats
3. **Text Processing**: Content is processed through the multi-provider system
4. **Chunking Strategy**: Large documents are split into manageable chunks
5. **AI Processing**: Each chunk is processed by available AI providers
6. **Result Assembly**: Processed chunks are reassembled into final output
7. **Output Delivery**: Results are presented to users with download options

## External Dependencies

### AI Providers
- **OpenAI**: GPT-4 models for text processing
- **Anthropic**: Claude models for advanced reasoning
- **Perplexity**: Research and factual content processing
- **DeepSeek**: High-performance alternative processing with full integration
- **Azure OpenAI**: Enterprise-grade OpenAI access

### Third-Party Services
- **Google Cloud TTS**: Text-to-speech conversion (legacy)
- **ElevenLabs**: High-quality voice synthesis
- **GPTZero**: AI content detection
- **SendGrid**: Email delivery service

### Processing Libraries
- **PyPDF2**: PDF text extraction
- **python-docx**: Word document processing
- **Pillow/pytesseract**: Image processing and OCR
- **SpeechRecognition**: Audio transcription
- **pydub**: Audio file manipulation
- **langdetect**: Language identification

## Deployment Strategy

### Development Environment
- Local Flask development server
- SQLite database for rapid prototyping
- Environment variables for API key management
- File-based uploads with configurable storage

### Production Considerations
- WSGI-compatible deployment (gunicorn recommended)
- PostgreSQL database for production workloads
- Redis for session management and caching
- Cloud storage for file uploads (AWS S3, Google Cloud Storage)
- Load balancing for high-traffic scenarios

### Security Features
- API key rotation and health monitoring
- File upload validation and sanitization
- Rate limiting per user and per API key
- Secure session management
- Input validation and sanitization

## Changelog
- July 29, 2025. CRITICAL FIX IMPLEMENTED: Dollar sign conversion system deployed across all text processing functions to prevent LaTeX formatting conflicts - converts "$13" to "13 dollars" automatically in rewrite, homework, and all expansion processes
- July 29, 2025. Continued successful operation confirmed - rewrite functionality stable, user satisfaction confirmed "BETTER"
- July 28, 2025. MAJOR SUCCESS: Rewrite functionality fully operational - text processing working with successful expansion (2,128 to 2,720+ words), MathJax rendering confirmed, multi-chunk processing stable
- July 28, 2025. DeepSeek AI provider fully integrated across all interface dropdowns and backend functions
- July 07, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.
User satisfaction: EXCELLENT - Multiple "BETTER" confirmations after successful rewrite functionality restoration and continued operation.

## Recent Status
- Text rewrite processing: FULLY OPERATIONAL ✓
- Mathematical notation rendering: WORKING ✓  
- Multi-chunk processing: STABLE ✓
- AI provider integration: COMPLETE ✓
- DeepSeek integration: FULLY DEPLOYED ✓
- Dollar sign conversion: IMPLEMENTED ✓ (Prevents LaTeX formatting conflicts)