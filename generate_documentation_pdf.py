#!/usr/bin/env python3
"""
Generate comprehensive PDF documentation for HC-Search-Models project
Requires: pip install reportlab pillow
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from datetime import datetime
import textwrap

def create_pdf():
    """Generate PDF documentation"""
    
    # Create PDF document
    filename = "HC-Search-Models_Documentation.pdf"
    doc = SimpleDocTemplate(filename, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=6,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#2e5c8a'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=13,
        textColor=colors.HexColor('#3d6fa3'),
        spaceAfter=10,
        spaceBefore=10,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=8
    )
    
    # ==================== PAGE 1: TITLE PAGE ====================
    elements.append(Spacer(1, 1.5*inch))
    
    title = Paragraph("HC-Search-Models", title_style)
    elements.append(title)
    
    elements.append(Spacer(1, 0.2*inch))
    
    subtitle = Paragraph("Healthcare Semantic Search System", 
                        ParagraphStyle('Subtitle', parent=styles['Normal'], 
                                     fontSize=16, alignment=TA_CENTER, 
                                     textColor=colors.HexColor('#666666')))
    elements.append(subtitle)
    
    elements.append(Spacer(1, 0.5*inch))
    
    # Project details
    details_data = [
        ["Repository:", "github.com/SumedhKolte/HC-Search-Models"],
        ["Repository ID:", "963410836"],
        ["Primary Language:", "Python (87.5%)"],
        ["Generated:", datetime.now().strftime("%B %d, %Y")],
    ]
    
    details_table = Table(details_data, colWidths=[2*inch, 3.5*inch])
    details_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1f4788')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    
    elements.append(details_table)
    elements.append(Spacer(1, 1*inch))
    
    overview = Paragraph(
        """<b>Overview:</b> This project implements an advanced semantic search system for healthcare 
        entities including doctors, hospitals, clinics, and diseases. It leverages transformer-based 
        embeddings and vector similarity search to provide intelligent, context-aware medical search 
        capabilities with sub-100ms response times.""",
        body_style
    )
    elements.append(overview)
    
    elements.append(PageBreak())
    
    # ==================== TABLE OF CONTENTS ====================
    elements.append(Paragraph("Table of Contents", heading1_style))
    
    toc_items = [
        "1. Project Overview",
        "2. Architecture Overview",
        "3. Model Training & Inference Pipeline",
        "4. Core Components & Technology Stack",
        "5. Search Pipeline - Detailed Flowchart",
        "6. Technology Stack Details",
        "7. Model Configuration",
        "8. Key Features",
        "9. Language Composition",
    ]
    
    for item in toc_items:
        elements.append(Paragraph(item, ParagraphStyle('TOC', parent=styles['Normal'], 
                                                       fontSize=10, leftIndent=20, spaceAfter=6)))
    
    elements.append(PageBreak())
    
    # ==================== PAGE 3: PROJECT OVERVIEW ====================
    elements.append(Paragraph("1. Project Overview", heading1_style))
    
    overview_text = """
    <b>HC-Search-Models</b> is a healthcare-focused semantic search system that combines state-of-the-art 
    natural language processing with vector similarity search. The system enables users to find doctors, 
    hospitals, clinics, and disease information through intelligent semantic search rather than simple 
    keyword matching.
    """
    elements.append(Paragraph(overview_text, body_style))
    elements.append(Spacer(1, 0.1*inch))
    
    elements.append(Paragraph("Purpose & Goals:", heading2_style))
    goals_text = """
    • Provide accurate, context-aware search for medical professionals and facilities<br/>
    • Enable semantic understanding of medical queries (e.g., "heart specialist" → cardiologist)<br/>
    • Support location-based filtering and proximity searches<br/>
    • Maintain high performance with sub-100ms response times<br/>
    • Track search patterns and continuously improve model accuracy<br/>
    • Ensure ACID compliance and data reliability
    """
    elements.append(Paragraph(goals_text, body_style))
    
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph("Key Statistics:", heading2_style))
    
    stats_data = [
        ["Metric", "Value"],
        ["Repository Language", "Python (87.5%)"],
        ["Embedding Dimension", "384"],
        ["Default Search Results", "10"],
        ["Max Search Results", "100"],
        ["Model Base", "all-MiniLM-L6-v2"],
        ["Default Training Epochs", "5"],
        ["Min Similarity Score", "0.5"],
    ]
    
    stats_table = Table(stats_data, colWidths=[2.5*inch, 2.5*inch])
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
    ]))
    
    elements.append(stats_table)
    elements.append(PageBreak())
    
    # ==================== PAGE 4: ARCHITECTURE ====================
    elements.append(Paragraph("2. Architecture Overview", heading1_style))
    
    arch_text = """
    The HC-Search-Models system is built on a multi-layered architecture that separates concerns 
    into distinct components. This modular design ensures scalability, maintainability, and 
    flexibility for future enhancements.
    """
    elements.append(Paragraph(arch_text, body_style))
    elements.append(Spacer(1, 0.15*inch))
    
    elements.append(Paragraph("Architecture Layers:", heading2_style))
    
    layers_text = """
    <b>1. User Interface Layer:</b> Accepts user queries and parameters<br/>
    <b>2. Validation Layer:</b> Validates input syntax, entity types, and constraints<br/>
    <b>3. Embedding Layer:</b> Converts text to 384-dimensional vectors using Sentence Transformers<br/>
    <b>4. Search Layer:</b> Performs similarity search using FAISS or PostgreSQL pgvector<br/>
    <b>5. Filtering Layer:</b> Applies business logic filters (location, specialization, ratings)<br/>
    <b>6. Ranking Layer:</b> Re-ranks results based on multiple criteria<br/>
    <b>7. Database Layer:</b> PostgreSQL with pgvector extension for vector storage<br/>
    <b>8. Monitoring Layer:</b> Tracks performance metrics and search patterns
    """
    elements.append(Paragraph(layers_text, body_style))
    
    elements.append(Spacer(1, 0.15*inch))
    
    elements.append(Paragraph("Data Flow:", heading2_style))
    
    flow_text = """
    1. User submits search query<br/>
    2. Query is validated and preprocessed<br/>
    3. Query embedding is generated<br/>
    4. FAISS index or PostgreSQL vector search executed<br/>
    5. Results filtered by user-provided criteria<br/>
    6. Results ranked by similarity and other metrics<br/>
    7. Top K results returned to user<br/>
    8. Search event logged for analytics
    """
    elements.append(Paragraph(flow_text, body_style))
    
    elements.append(PageBreak())
    
    # ==================== PAGE 5: MODEL TRAINING ====================
    elements.append(Paragraph("3. Model Training & Inference Pipeline", heading1_style))
    
    elements.append(Paragraph("Training Phase:", heading2_style))
    
    training_text = """
    <b>Data Collection:</b> The system collects data from multiple sources:<br/>
    • Medical professional profiles (doctors, specialists)<br/>
    • Healthcare facility information (hospitals, clinics)<br/>
    • Disease and symptom datasets<br/>
    • User search patterns and feedback<br/>
    <br/>
    <b>Data Preprocessing:</b> Raw data undergoes multiple transformation steps:<br/>
    • Text normalization and tokenization<br/>
    • Query expansion using synonym detection<br/>
    • Full-text search vector creation (PostgreSQL tsvector)<br/>
    • Handling of special characters and abbreviations<br/>
    <br/>
    <b>Embedding Generation:</b> All text is converted to vectors:<br/>
    • Using Sentence Transformers (all-MiniLM-L6-v2)<br/>
    • Creates 384-dimensional embeddings<br/>
    • Embeddings stored in PostgreSQL and FAISS<br/>
    <br/>
    <b>Fine-tuning (Optional):</b> Domain-specific model optimization:<br/>
    • CosineSimilarityLoss for semantic similarity<br/>
    • Training for 5-10 epochs<br/>
    • Batch size: 32 samples<br/>
    • Warmup steps: 50, Evaluation every 100 steps
    """
    elements.append(Paragraph(training_text, body_style))
    
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph("Inference Phase:", heading2_style))
    
    inference_text = """
    During inference (search time):<br/>
    1. User query is embedded using the same trained model<br/>
    2. FAISS index performs approximate nearest neighbor search (O(log n) complexity)<br/>
    3. Alternatively, PostgreSQL pgvector performs exact or approximate search<br/>
    4. Top candidates are retrieved and post-processed<br/>
    5. Filters are applied to narrow results<br/>
    6. Final results ranked by similarity scores<br/>
    7. Response returned with execution metrics
    """
    elements.append(Paragraph(inference_text, body_style))
    
    elements.append(PageBreak())
    
    # ==================== PAGE 6: CORE COMPONENTS ====================
    elements.append(Paragraph("4. Core Components & Technology Stack", heading1_style))
    
    elements.append(Paragraph("ML/AI Layer:", heading2_style))
    
    ml_text = """
    <b>Embedding Models:</b><br/>
    • Base Model: Sentence Transformers (all-MiniLM-L6-v2)<br/>
    • Output Dimension: 384<br/>
    • Inference Speed: 10-20ms per query<br/>
    • Domain: General NLP with good healthcare understanding<br/>
    <br/>
    <b>Vector Indexing:</b><br/>
    • Primary: FAISS (Facebook AI Similarity Search)<br/>
    &nbsp;&nbsp;- Index Type: IVFFlat (Inverted File with Flat L2)<br/>
    &nbsp;&nbsp;- Query Complexity: O(log n)<br/>
    &nbsp;&nbsp;- Supports up to billions of vectors<br/>
    • Secondary: PostgreSQL pgvector<br/>
    &nbsp;&nbsp;- HNSW indexing option<br/>
    &nbsp;&nbsp;- ACID compliance and transactions<br/>
    &nbsp;&nbsp;- Native SQL integration
    """
    elements.append(Paragraph(ml_text, body_style))
    
    elements.append(Spacer(1, 0.15*inch))
    elements.append(Paragraph("Data Processing Layer:", heading2_style))
    
    data_text = """
    <b>Text Processing:</b><br/>
    • Query expansion using synonym detection<br/>
    • Spelling correction and normalization<br/>
    • Named Entity Recognition (NER)<br/>
    • Full-text search with PostgreSQL tsvector<br/>
    • Multi-language support<br/>
    <br/>
    <b>Database:</b><br/>
    • PostgreSQL 13+ with pgvector extension<br/>
    • Entities: doctors, hospitals, clinics, diseases, symptoms<br/>
    • Logging: search_logs, model_training_metrics, query_expansion_logs<br/>
    • Connection Pool: Min 1, Max 10 connections<br/>
    • SSL/TLS support for AWS RDS
    """
    elements.append(Paragraph(data_text, body_style))
    
    elements.append(PageBreak())
    
    # ==================== PAGE 7: SEARCH ENGINE ====================
    elements.append(Paragraph("Search Engine Layer (continued):", heading1_style))
    
    engine_text = """
    <b>Engine Components:</b><br/>
    • Query Validator: Syntax and semantic validation<br/>
    • Vector Generator: Embedding creation from text<br/>
    • FAISS Index Manager: Index loading and management<br/>
    • Results Aggregator: Combining results from multiple sources<br/>
    • Performance Metrics Tracker: Real-time monitoring<br/>
    <br/>
    <b>Search Features:</b><br/>
    • Multi-entity search (doctors, hospitals, diseases simultaneously)<br/>
    • Location-based filtering with radius-based search<br/>
    • Semantic similarity matching (cosine and L2 distances)<br/>
    • Result ranking by multiple criteria<br/>
    • Query logging and analytics<br/>
    • Result caching for frequently searched queries<br/>
    • Support for specialized searches (e.g., "Dr. + specialization")
    """
    elements.append(Paragraph(engine_text, body_style))
    
    elements.append(PageBreak())
    
    # ==================== PAGE 8: SEARCH PIPELINE ====================
    elements.append(Paragraph("5. Search Pipeline - Detailed Flowchart", heading1_style))
    
    pipeline_text = """
    The search pipeline executes in the following stages:
    """
    elements.append(Paragraph(pipeline_text, body_style))
    elements.append(Spacer(1, 0.1*inch))
    
    pipeline_steps = """
    <b>Stage 1: Validation</b><br/>
    • Query length validation (max 128 tokens)<br/>
    • Character set validation<br/>
    • Language detection<br/>
    • Entity type detection (doctor/hospital/disease)<br/>
    <br/>
    <b>Stage 2: Preprocessing</b><br/>
    • Convert to lowercase<br/>
    • Handle special characters<br/>
    • Expand abbreviations (Dr. → Doctor)<br/>
    • Expand query with synonyms<br/>
    <br/>
    <b>Stage 3: Embedding</b><br/>
    • Use Sentence Transformers to create 384-dim vector<br/>
    • Normalize embedding<br/>
    • Optionally apply dimensionality reduction<br/>
    <br/>
    <b>Stage 4: Vector Search</b><br/>
    • Query FAISS index for approximate nearest neighbors (top 100)<br/>
    • Or query PostgreSQL pgvector for exact/approximate search<br/>
    • Calculate similarity scores<br/>
    <br/>
    <b>Stage 5: Basic Filtering</b><br/>
    • Apply city/location filters<br/>
    • Apply specialization filters<br/>
    • Filter by entity status (active/inactive)<br/>
    <br/>
    <b>Stage 6: Business Logic Filtering</b><br/>
    • Apply rating threshold<br/>
    • Filter by experience level<br/>
    • Check availability windows<br/>
    • Apply fee range constraints<br/>
    <br/>
    <b>Stage 7: Ranking</b><br/>
    • Primary: Similarity score<br/>
    • Secondary: Entity rating<br/>
    • Tertiary: Relevance boost<br/>
    • Optional: Geographic distance<br/>
    <br/>
    <b>Stage 8: Post-processing</b><br/>
    • Limit results (default 10, max 100)<br/>
    • Format response data<br/>
    • Calculate execution metrics<br/>
    • Log search event for analytics<br/>
    <br/>
    <b>Stage 9: Return</b><br/>
    • Result items with scores<br/>
    • Total result count<br/>
    • Applied filters information<br/>
    • Execution time
    """
    elements.append(Paragraph(pipeline_steps, body_style))
    
    elements.append(PageBreak())
    
    # ==================== PAGE 9: TECHNOLOGY STACK ====================
    elements.append(Paragraph("6. Technology Stack Details", heading1_style))
    
    tech_stack_data = [
        ["Layer", "Technology", "Purpose"],
        ["ML Framework", "PyTorch + Transformers", "Neural network foundation"],
        ["Embeddings", "Sentence-Transformers", "Text-to-vector conversion"],
        ["Vector Search", "FAISS + pgvector", "Similarity search & indexing"],
        ["Database", "PostgreSQL 13+", "Data storage & vector storage"],
        ["Language", "Python 87.5%", "Main implementation"],
        ["Optimization", "Optuna", "Hyperparameter tuning"],
        ["Logging", "Python logging + JSON", "Performance tracking"],
        ["Geolocation", "GeoPy", "Location-based search"],
        ["Data Science", "NumPy, Pandas, SKLearn", "Data manipulation & ML"],
        ["ORM", "SQLAlchemy", "Database abstraction"],
        ["Low-level", "C + Perl", "Performance-critical ops"],
    ]
    
    tech_table = Table(tech_stack_data, colWidths=[1.8*inch, 1.8*inch, 1.8*inch])
    tech_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    
    elements.append(tech_table)
    
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph("Key Dependencies:", heading2_style))
    
    deps_text = """
    <b>Core ML Dependencies:</b><br/>
    • sentence-transformers >= 2.2.2<br/>
    • torch >= 1.13.0<br/>
    • transformers >= 4.25.0<br/>
    • faiss-cpu >= 1.7.4<br/>
    <br/>
    <b>Data Processing:</b><br/>
    • numpy >= 1.24.3<br/>
    • pandas >= 2.1.3<br/>
    • scikit-learn >= 1.3.2<br/>
    <br/>
    <b>Database:</b><br/>
    • sqlalchemy >= 2.0.23<br/>
    • psycopg2-binary >= 2.9.9<br/>
    <br/>
    <b>Utilities:</b><br/>
    • python-dotenv >= 1.0.0<br/>
    • requests >= 2.31.0<br/>
    • geopy >= 2.4.1
    """
    elements.append(Paragraph(deps_text, body_style))
    
    elements.append(PageBreak())
    
    # ==================== PAGE 10: MODEL CONFIGURATION ====================
    elements.append(Paragraph("7. Model Configuration", heading1_style))
    
    config_text = """
    The system uses a comprehensive configuration file that defines all model parameters, 
    database settings, and search behavior. Below are the key configuration values:
    """
    elements.append(Paragraph(config_text, body_style))
    
    elements.append(Spacer(1, 0.15*inch))
    
    config_data = [
        ["Parameter", "Value", "Description"],
        ["base_model", "all-MiniLM-L6-v2", "Pre-trained embedding model"],
        ["embedding_dim", "384", "Vector dimension size"],
        ["batch_size", "32", "Batch size for encoding"],
        ["max_seq_length", "128", "Maximum query token length"],
        ["epochs", "5", "Training epochs"],
        ["evaluation_steps", "100", "Steps between evaluations"],
        ["warmup_steps", "50", "Learning rate warmup steps"],
        ["default_limit", "10", "Default search results count"],
        ["min_similarity_score", "0.5", "Minimum similarity threshold"],
        ["max_expansion_terms", "5", "Max query expansion terms"],
        ["location_radius_km", "10.0", "Default search radius"],
        ["cache_results", "true", "Enable result caching"],
    ]
    
    config_table = Table(config_data, colWidths=[1.7*inch, 1.5*inch, 2.3*inch])
    config_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.lightgrey, colors.lightyellow]),
    ]))
    
    elements.append(config_table)
    
    elements.append(Spacer(1, 0.15*inch))
    elements.append(Paragraph("Entity Configuration:", heading2_style))
    
    entity_text = """
    Each entity type (doctors, hospitals, etc.) has its own configuration:<br/>
    <br/>
    <b>Doctors:</b><br/>
    • Table: doctors | ID Field: did<br/>
    • Searchable Fields: name, specialization, city, degree<br/>
    • Filterable Fields: specialization, rating, city, experience<br/>
    <br/>
    <b>Hospitals:</b><br/>
    • Table: hospitals | ID Field: hid<br/>
    • Searchable Fields: name, hospital_type, location<br/>
    • Filterable Fields: hospital_type, rating, location<br/>
    <br/>
    <b>Diseases:</b><br/>
    • Table: diseases | ID Field: disease_id<br/>
    • Searchable Fields: name, common_names, description<br/>
    • Filterable Fields: tags
    """
    elements.append(Paragraph(entity_text, body_style))
    
    elements.append(PageBreak())
    
    # ==================== PAGE 11: KEY FEATURES ====================
    elements.append(Paragraph("8. Key Features", heading1_style))
    
    features_data = [
        ["Feature", "Description", "Benefit"],
        ["Semantic Search", "Understands meaning beyond keywords", "Better search accuracy"],
        ["Multi-Entity", "Search across doctors, hospitals, diseases", "Comprehensive results"],
        ["Location-Based", "Geographic proximity filtering", "Localized results"],
        ["High Performance", "Sub-100ms response times", "Real-time user experience"],
        ["Scalable", "FAISS for millions of entities", "Handles growth"],
        ["ACID Compliance", "PostgreSQL reliability", "Data integrity"],
        ["Continuous Learning", "Trains on search patterns", "Improving accuracy"],
        ["Metrics Tracking", "Performance monitoring built-in", "System visibility"],
        ["Query Expansion", "Synonym and abbreviation handling", "Broader matching"],
        ["Result Ranking", "Multi-criteria ranking", "Best results first"],
        ["Caching", "Result caching for common queries", "Faster responses"],
    ]
    
    features_table = Table(features_data, colWidths=[1.5*inch, 2*inch, 1.8*inch])
    features_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.lightgrey, colors.lightyellow]),
    ]))
    
    elements.append(features_table)
    
    elements.append(PageBreak())
    
    # ==================== PAGE 12: LANGUAGE COMPOSITION ====================
    elements.append(Paragraph("9. Language Composition", heading1_style))
    
    comp_text = """
    The HC-Search-Models project is primarily written in Python with performance-critical 
    components in C and supporting utilities in Perl and Jupyter Notebooks.
    """
    elements.append(Paragraph(comp_text, body_style))
    
    elements.append(Spacer(1, 0.15*inch))
    
    comp_data = [
        ["Language", "Percentage", "Purpose"],
        ["Python", "87.5%", "Core ML, search engine, data processing"],
        ["C", "8.2%", "FAISS vector indexing performance"],
        ["Perl", "2.4%", "Database utilities and scripts"],
        ["Jupyter Notebook", "1.8%", "Model experimentation and training"],
        ["Other", "0.1%", "Configuration and misc files"],
    ]
    
    comp_table = Table(comp_data, colWidths=[1.5*inch, 1.5*inch, 3*inch])
    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (1, -1), 'CENTER'),
        ('ALIGN', (2, 0), (2, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.lightgrey, colors.lightyellow]),
    ]))
    
    elements.append(comp_table)
    
    elements.append(Spacer(1, 0.3*inch))
    
    elements.append(Paragraph("Component Breakdown:", heading2_style))
    
    breakdown_text = """
    <b>Python (87.5%) - Core Implementation:</b><br/>
    • Search engine logic and query processing<br/>
    • Embedding generation and model management<br/>
    • Database abstraction and ORM<br/>
    • REST API and web service layer<br/>
    • Data pipeline and preprocessing<br/>
    • Performance metrics and logging<br/>
    <br/>
    <b>C (8.2%) - Performance Layer:</b><br/>
    • FAISS vector index compilation<br/>
    • Low-level similarity computations<br/>
    • Memory-optimized data structures<br/>
    <br/>
    <b>Perl (2.4%) - Utilities:</b><br/>
    • Database migration scripts<br/>
    • Data import/export utilities<br/>
    • System maintenance scripts<br/>
    <br/>
    <b>Jupyter Notebooks (1.8%) - Experimentation:</b><br/>
    • Model training workflows<br/>
    • Data exploration and analysis<br/>
    • Performance benchmarking<br/>
    • Documentation and tutorials
    """
    elements.append(Paragraph(breakdown_text, body_style))
    
    elements.append(PageBreak())
    
    # ==================== FINAL PAGE: CONCLUSION ====================
    elements.append(Paragraph("Conclusion", heading1_style))
    
    conclusion_text = """
    The HC-Search-Models project represents a sophisticated approach to healthcare search 
    by combining semantic understanding with high-performance vector similarity search. 
    The system's architecture is designed to be scalable, maintainable, and user-friendly.
    <br/><br/>
    <b>Key Strengths:</b><br/>
    • Semantic understanding beyond keyword matching<br/>
    • High-performance vector search with FAISS and pgvector<br/>
    • Flexible architecture supporting multiple entity types<br/>
    • Comprehensive logging and metrics tracking<br/>
    • Domain-optimized embeddings for healthcare<br/>
    • ACID-compliant database with advanced vector indexing<br/>
    <br/>
    <b>Future Enhancements:</b><br/>
    • Real-time model updating from user feedback<br/>
    • Multi-language support expansion<br/>
    • Advanced query understanding with NER<br/>
    • Personalized ranking based on user history<br/>
    • Integration with recommendation systems<br/>
    • Mobile application support<br/>
    <br/>
    This system provides a solid foundation for building intelligent healthcare search 
    applications with state-of-the-art semantic understanding and performance.
    """
    elements.append(Paragraph(conclusion_text, body_style))
    
    elements.append(Spacer(1, 0.5*inch))
    
    footer_text = f"""
    <i>Document Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</i><br/>
    <i>Repository: github.com/SumedhKolte/HC-Search-Models</i><br/>
    <i>For more information, visit the repository on GitHub</i>
    """
    elements.append(Paragraph(footer_text, 
                            ParagraphStyle('Footer', parent=styles['Normal'], 
                                         fontSize=8, alignment=TA_CENTER, 
                                         textColor=colors.grey)))
    
    # ==================== BUILD PDF ====================
    doc.build(elements)
    print(f"✓ PDF generated successfully: {filename}")
    return filename

if __name__ == "__main__":
    create_pdf()
