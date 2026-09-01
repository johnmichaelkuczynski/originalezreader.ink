from datetime import datetime
from app import db

class UserProfile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)  # Email used as unique identifier
    merged_text = db.Column(db.Text, nullable=True)  # Combined text from all uploads
    word_count = db.Column(db.Integer, default=0)  # Total word count of merged text
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    uploads = db.relationship('WritingSample', backref='profile', lazy=True)
    entries = db.relationship('TextEntry', backref='user_profile', lazy=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'word_count': self.word_count,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'upload_count': len(self.uploads)
        }

class WritingSample(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    profile_id = db.Column(db.Integer, db.ForeignKey('user_profile.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    text_content = db.Column(db.Text, nullable=False)
    word_count = db.Column(db.Integer, default=0)
    file_type = db.Column(db.String(20), nullable=False)  # 'txt', 'pdf', 'docx'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'profile_id': self.profile_id,
            'filename': self.filename,
            'text_content': self.text_content,
            'word_count': self.word_count,
            'file_type': self.file_type,
            'created_at': self.created_at.isoformat()
        }

class ContentSource(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    text_content = db.Column(db.Text, nullable=False)
    word_count = db.Column(db.Integer, default=0)
    file_type = db.Column(db.String(20), nullable=False)  # 'txt', 'pdf', 'docx', etc.
    usage_instructions = db.Column(db.Text, nullable=True)  # How the content source should be used
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship with TextEntry (nullable so it can be used before having a text entry)
    text_entry_id = db.Column(db.Integer, db.ForeignKey('text_entry.id'), nullable=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'text_content': self.text_content,
            'word_count': self.word_count,
            'file_type': self.file_type,
            'usage_instructions': self.usage_instructions,
            'created_at': self.created_at.isoformat(),
            'text_entry_id': self.text_entry_id
        }

class TextEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    original_text = db.Column(db.Text, nullable=False)
    processed_text = db.Column(db.Text, nullable=False)
    action = db.Column(db.String(50), nullable=False)  # rewrite, summarize, expand
    complexity = db.Column(db.String(100), nullable=False)  # Increased from 20 to 100 characters
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    # Add total chunks field to track pagination
    total_chunks = db.Column(db.Integer, default=1)
    # Add new fields for custom processing
    custom_instructions = db.Column(db.Text)
    preserve_structure = db.Column(db.Boolean, default=True)
    # User's writing profile ID (optional)
    user_profile_id = db.Column(db.Integer, db.ForeignKey('user_profile.id'), nullable=True)
    # Language translation support
    target_language = db.Column(db.String(50), nullable=True)  # Target language for translation
    # Add chunks relationship
    chunks = db.relationship('DocumentChunk', backref='document', lazy=True)
    # Add content source relationship
    content_sources = db.relationship('ContentSource', backref='text_entry', lazy=True)

    def to_dict(self):
        return {
            'id': self.id,
            'original_text': self.original_text,
            'processed_text': self.processed_text,
            'action': self.action,
            'complexity': self.complexity,
            'created_at': self.created_at.isoformat(),
            'total_chunks': self.total_chunks,
            'current_chunk': min(len(self.chunks), self.total_chunks),
            'custom_instructions': self.custom_instructions,
            'preserve_structure': self.preserve_structure,
            'user_profile_id': self.user_profile_id,
            'target_language': self.target_language,
            'has_content_source': len(self.content_sources) > 0
        }

class DocumentChunk(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey('text_entry.id'), nullable=False)
    chunk_number = db.Column(db.Integer, nullable=False)  # Position in document
    original_chunk = db.Column(db.Text, nullable=False)  # Original text chunk
    processed_chunk = db.Column(db.Text)  # Processed version (nullable until processed)
    is_processed = db.Column(db.Boolean, default=False)
    processing_status = db.Column(db.String(20), default='pending')  # pending, processing, complete, error
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'document_id': self.document_id,
            'chunk_number': self.chunk_number,
            'original_chunk': self.original_chunk,
            'processed_chunk': self.processed_chunk,
            'is_processed': self.is_processed,
            'processing_status': self.processing_status,
            'created_at': self.created_at.isoformat()
        }

class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    message = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    context = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'message': self.message,
            'response': self.response,
            'context': self.context,
            'created_at': self.created_at.isoformat()
        }


# Long-document coherence state.  These tables are deliberately separate from
# TextEntry/DocumentChunk so an interrupted coherence run can be resumed and
# audited without changing the legacy chunk-processing path.
class CoherenceJob(db.Model):
    __tablename__ = 'coherence_jobs'

    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey('text_entry.id'), nullable=False, index=True)
    user_id = db.Column(db.String(255), nullable=True, index=True)
    document_title = db.Column(db.String(512), nullable=True)
    original_text = db.Column(db.Text, nullable=False)
    total_input_words = db.Column(db.Integer, default=0)
    target_min_words = db.Column(db.Integer, nullable=True)
    target_max_words = db.Column(db.Integer, nullable=True)
    target_mid_words = db.Column(db.Integer, nullable=True)
    length_ratio = db.Column(db.Float, nullable=True)
    length_mode = db.Column(db.String(32), nullable=True)
    num_chunks = db.Column(db.Integer, default=0)
    chunk_target_words = db.Column(db.Integer, nullable=True)
    global_skeleton = db.Column(db.JSON, nullable=True)
    custom_instructions = db.Column(db.Text, nullable=True)
    author_style = db.Column(db.String(255), nullable=True)
    style_source = db.Column(db.Text, nullable=True)
    content_source = db.Column(db.Text, nullable=True)
    provider_preference = db.Column(db.String(32), nullable=True)
    selected_chunks = db.Column(db.JSON, nullable=True)
    skeleton_cursor = db.Column(db.Integer, default=0)
    skeleton_sections = db.Column(db.JSON, nullable=True)
    validation_report = db.Column(db.JSON, nullable=True)
    warnings = db.Column(db.JSON, nullable=True)
    repair_cursor = db.Column(db.Integer, default=0)
    status = db.Column(db.String(32), default='pending', index=True)
    current_chunk = db.Column(db.Integer, default=0)
    final_output = db.Column(db.Text, nullable=True)
    final_word_count = db.Column(db.Integer, nullable=True)
    error_message = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    chunks = db.relationship('CoherenceChunk', backref='coherence_job',
                             lazy=True, cascade='all, delete-orphan')
    runs = db.relationship('CoherenceRun', backref='coherence_job',
                           lazy=True, cascade='all, delete-orphan')


class CoherenceChunk(db.Model):
    __tablename__ = 'coherence_chunks'
    __table_args__ = (db.UniqueConstraint('job_id', 'chunk_index',
                                          name='uq_coherence_chunk_position'),)

    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.Integer, db.ForeignKey('coherence_jobs.id'),
                       nullable=False, index=True)
    chunk_index = db.Column(db.Integer, nullable=False)
    chunk_input_text = db.Column(db.Text, nullable=False)
    chunk_input_words = db.Column(db.Integer, default=0)
    target_words = db.Column(db.Integer, nullable=True)
    min_words = db.Column(db.Integer, nullable=True)
    max_words = db.Column(db.Integer, nullable=True)
    chunk_output_text = db.Column(db.Text, nullable=True)
    actual_words = db.Column(db.Integer, nullable=True)
    chunk_delta = db.Column(db.JSON, nullable=True)
    retry_count = db.Column(db.Integer, default=0)
    status = db.Column(db.String(32), default='pending', index=True)
    error_message = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TractatusTier(db.Model):
    __tablename__ = 'tractatus_tiers'
    __table_args__ = (db.UniqueConstraint('job_id', 'job_type', 'tier',
                                          name='uq_tractatus_tier'),)

    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.Integer, db.ForeignKey('coherence_jobs.id'),
                       nullable=False, index=True)
    job_type = db.Column(db.String(100), nullable=False)
    tier = db.Column(db.Integer, nullable=False)
    tree = db.Column(db.JSON, nullable=False, default=dict)
    node_count = db.Column(db.Integer, nullable=False, default=0)
    parent_tier_id = db.Column(db.Integer, db.ForeignKey('tractatus_tiers.id'),
                               nullable=True)
    compression_count = db.Column(db.Integer, nullable=False, default=0)
    last_update = db.Column(db.DateTime, default=datetime.utcnow,
                            onupdate=datetime.utcnow, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class TractatusArchive(db.Model):
    __tablename__ = 'tractatus_archive'

    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.Integer, db.ForeignKey('coherence_jobs.id'),
                       nullable=False, index=True)
    job_type = db.Column(db.String(100), nullable=False, index=True)
    tier = db.Column(db.Integer, nullable=False)
    tree_snapshot = db.Column(db.JSON, nullable=False)
    node_count_at_snapshot = db.Column(db.Integer, nullable=False)
    reason = db.Column(db.String(32), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)


class CoherenceRun(db.Model):
    __tablename__ = 'coherence_runs'

    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.Integer, db.ForeignKey('coherence_jobs.id'),
                       nullable=False, index=True)
    run_type = db.Column(db.String(32), nullable=False, index=True)
    chunk_index = db.Column(db.Integer, nullable=True)
    run_input = db.Column(db.JSON, nullable=True)
    run_output = db.Column(db.JSON, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)