from setuptools import setup, find_packages
from pathlib import Path
import os
from datetime import datetime, timezone
import requests

# SSL Certificate Setup
def setup_ssl():
    """Setup SSL certificate for AWS RDS"""
    ssl_dir = Path(__file__).parent / 'ssl'
    ssl_dir.mkdir(exist_ok=True)
    cert_path = ssl_dir / 'rds-ca-bundle.pem'
    timestamp_path = ssl_dir / 'cert_timestamp.txt'
    
    needs_update = True
    if cert_path.exists() and timestamp_path.exists():
        with open(timestamp_path, 'r') as f:
            last_update = datetime.fromisoformat(f.read().strip())
            if (datetime.now(timezone.utc) - last_update).days < 30:
                needs_update = False
    
    if needs_update:
        print(f"Downloading SSL certificate at {datetime.now(timezone.utc).isoformat()}")
        response = requests.get('https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem')
        response.raise_for_status()
        
        with open(cert_path, 'wb') as f:
            f.write(response.content)
        
        with open(timestamp_path, 'w') as f:
            f.write(datetime.now(timezone.utc).isoformat())
    
    return str(cert_path)

# Ensure SSL certificate is present
SSL_CERT_PATH = setup_ssl()

# Write requirements files if they don't exist
requirements = {
    'base.txt': [
        'numpy>=1.24.3',
        'pandas>=2.1.3',
        'sqlalchemy>=2.0.23',
        'psycopg2-binary>=2.9.9',
        'faiss-cpu>=1.7.4',
        'sentence-transformers>=2.2.2',
        'python-dotenv>=1.0.0',
        'requests>=2.31.0',
        'python-json-logger>=2.0.7',
        'scikit-learn>=1.3.2',
        'geopy>=2.4.1'
    ],
    'dev.txt': [
        'numpy>=1.24.3',
        'pandas>=2.1.3',
        'sqlalchemy>=2.0.23',
        'psycopg2-binary>=2.9.9',
        'faiss-cpu>=1.7.4',
        'sentence-transformers>=2.2.2',
        'python-dotenv>=1.0.0',
        'requests>=2.31.0',
        'python-json-logger>=2.0.7',
        'scikit-learn>=1.3.2',
        'geopy>=2.4.1',
        'pytest>=7.4.0',
        'pytest-cov>=4.1.0',
        'pytest-asyncio>=0.21.1',
        'black>=23.11.0',
        'isort>=5.12.0',
        'mypy>=1.7.0',
        'jupyter>=1.0.0'
    ],
    'prod.txt': [
        'numpy>=1.24.3',
        'pandas>=2.1.3',
        'sqlalchemy>=2.0.23',
        'psycopg2-binary>=2.9.9',
        'faiss-cpu>=1.7.4',
        'sentence-transformers>=2.2.2',
        'python-dotenv>=1.0.0',
        'requests>=2.31.0',
        'python-json-logger>=2.0.7',
        'scikit-learn>=1.3.2',
        'geopy>=2.4.1',
        'gunicorn>=21.2.0',
        'uvicorn>=0.24.0',
        'supervisor>=4.2.5'
    ]
}

requirements_dir = Path(__file__).parent / 'requirements'
requirements_dir.mkdir(exist_ok=True)

# Write requirements files
for filename, packages in requirements.items():
    req_file = requirements_dir / filename
    if not req_file.exists():
        with open(req_file, 'w') as f:
            f.write('\n'.join(packages))

# Read requirements for setup
def read_requirements(filename):
    req_file = Path(__file__).parent / 'requirements' / filename
    if req_file.exists():
        with open(req_file) as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="medical_search",
    version="1.0.0",
    description="Medical search system with vector and text-based search capabilities",
    author="Medical Search Team",
    author_email="team@medicalsearch.com",
    
    # Package configuration
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Data files
    package_data={
        "medical_search": [
            "ml/models/*",
            "ml/indexes/*",
            "ml/embeddings/*",
            "config/*.json",
            "ssl/*"
        ]
    },
    
    # Dependencies
    install_requires=read_requirements('base.txt'),
    extras_require={
        'dev': read_requirements('dev.txt'),
        'prod': read_requirements('prod.txt'),
    },
    
    # Python version
    python_requires=">=3.8",
    
    # Entry points
    entry_points={
        'console_scripts': [
            'medical-search=src.core.engine:main',
            'update-indexes=scripts.update_indexes:main',
            'train-model=scripts.train:main',
            'evaluate=scripts.evaluate:main'
        ]
    },
    
    # Classifiers
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Healthcare Industry',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    
    # Project URLs
    project_urls={
        'Source': 'https://github.com/username/medical-search',
        'Documentation': 'https://medical-search.readthedocs.io/',
    }
)