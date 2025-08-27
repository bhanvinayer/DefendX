# DefendX Installation Guide

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 500MB free space
- **Network**: Internet connection for initial package installation

### Recommended Requirements
- **Python**: 3.9 or 3.10
- **RAM**: 8GB or more
- **Storage**: 1GB free space
- **Processor**: Multi-core CPU for better performance

## Installation Methods

### Method 1: Standard Installation (Recommended)

1. **Clone or Download the Repository**
   ```bash
   git clone <repository-url>
   cd defendx
   ```

2. **Create Virtual Environment (Recommended)**
   ```bash
   # Create virtual environment
   python -m venv defendx-env
   
   # Activate virtual environment
   # On Windows:
   defendx-env\Scripts\activate
   
   # On macOS/Linux:
   source defendx-env/bin/activate
   ```

3. **Install Required Packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   ```bash
   streamlit run src/app.py
   ```

### Method 2: Development Installation

For developers who want to contribute or modify the system:

1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd defendx
   ```

2. **Install in Development Mode**
   ```bash
   pip install -e .
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

3. **Set up Pre-commit Hooks** (Optional)
   ```bash
   pre-commit install
   ```

## Package Dependencies

### Core Dependencies

The system requires the following packages (from `requirements.txt`):

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.3.0
plotly>=5.15.0
seaborn>=0.11.0
matplotlib>=3.5.0
joblib>=1.3.0
```

### Complete Requirements File

```txt
# Web Framework
streamlit==1.28.0

# Data Processing
pandas==2.1.1
numpy==1.25.2

# Machine Learning
scikit-learn==1.3.0
joblib==1.3.2

# Visualization
plotly==5.17.0
seaborn==0.12.2
matplotlib==3.7.2

# Additional utilities
typing-extensions>=4.0.0
```

## Configuration

### Environment Setup

1. **Create Data Directory Structure**
   ```bash
   mkdir -p data/models
   mkdir -p data/users
   ```

2. **Set Environment Variables** (Optional)
   ```bash
   # Set custom data directory
   export DEFENDX_DATA_DIR="/path/to/data"
   
   # Set custom port for Streamlit
   export STREAMLIT_PORT=8501
   ```

### Application Configuration

The application can be configured through Streamlit's configuration system:

1. **Create Streamlit Config Directory**
   ```bash
   mkdir -p ~/.streamlit
   ```

2. **Create Configuration File**
   ```bash
   # ~/.streamlit/config.toml
   [server]
   port = 8501
   headless = false
   
   [theme]
   base = "light"
   primaryColor = "#667eea"
   backgroundColor = "#ffffff"
   ```

## Running the Application

### Standard Startup

```bash
# Activate virtual environment (if using)
source defendx-env/bin/activate  # macOS/Linux
# or
defendx-env\Scripts\activate     # Windows

# Run the application
streamlit run src/app.py
```

### Custom Configuration

```bash
# Run on specific port
streamlit run src/app.py --server.port 8502

# Run in headless mode
streamlit run src/app.py --server.headless true

# Run with custom theme
streamlit run src/app.py --theme.base dark
```

### Production Deployment

For production deployment, consider:

```bash
# Install production dependencies
pip install gunicorn

# Run with production server (if applicable)
streamlit run src/app.py --server.headless true --server.enableCORS false
```

## Verification Steps

### 1. Package Installation Verification

```python
# test_installation.py
import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import plotly
import joblib

print("All packages imported successfully!")
print(f"Streamlit version: {st.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
```

### 2. Application Startup Verification

1. Start the application:
   ```bash
   streamlit run src/app.py
   ```

2. Open browser to `http://localhost:8501`

3. Verify all pages load:
   - Home page with statistics
   - Registration page
   - Verification page
   - Admin dashboard
   - Settings page

### 3. Functionality Testing

1. **Theme System**: Toggle between light and dark modes
2. **Registration**: Create a test user profile
3. **Data Persistence**: Check that `data/` directory is created
4. **Model Training**: Complete multiple registration sessions

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ModuleNotFoundError`
```
ModuleNotFoundError: No module named 'streamlit'
```

**Solution**:
```bash
# Ensure virtual environment is activated
source defendx-env/bin/activate  # macOS/Linux
defendx-env\Scripts\activate     # Windows

# Reinstall requirements
pip install -r requirements.txt
```

#### 2. Port Already in Use

**Problem**: 
```
OSError: [Errno 48] Address already in use
```

**Solution**:
```bash
# Use different port
streamlit run src/app.py --server.port 8502

# Or kill existing process
lsof -ti:8501 | xargs kill -9  # macOS/Linux
netstat -ano | findstr :8501   # Windows
```

#### 3. Permission Errors

**Problem**: Permission denied when creating data directories

**Solution**:
```bash
# Check current directory permissions
ls -la

# Create data directory with proper permissions
mkdir -p data && chmod 755 data
```

#### 4. Python Version Issues

**Problem**: Syntax errors or compatibility issues

**Solution**:
```bash
# Check Python version
python --version

# Use specific Python version
python3.9 -m venv defendx-env
```

### Performance Issues

#### Slow Startup

1. **Check System Resources**:
   ```bash
   # Monitor CPU and memory usage
   top          # macOS/Linux
   taskmgr      # Windows
   ```

2. **Optimize Virtual Environment**:
   ```bash
   # Use lighter package versions
   pip install --no-cache-dir -r requirements.txt
   ```

#### Memory Usage

1. **Monitor Memory Usage**:
   ```python
   import psutil
   print(f"Memory usage: {psutil.virtual_memory().percent}%")
   ```

2. **Optimize Data Loading**:
   - Limit session log size
   - Regular cleanup of old data
   - Use data compression

### Data Directory Issues

#### Missing Data Directory

```bash
# Create required directories
mkdir -p data/models
mkdir -p data/users
touch data/session_log.csv
```

#### Corrupted Data Files

```bash
# Backup existing data
cp data/session_log.csv data/session_log.csv.backup

# Reset data (if necessary)
rm data/session_log.csv
touch data/session_log.csv
```

## Advanced Configuration

### Custom Data Directory

```python
# Set custom data directory in application
import os
os.environ['DEFENDX_DATA_DIR'] = '/custom/path/to/data'
```

### Database Integration (Future)

For production environments, consider integrating with databases:

```python
# Example configuration for future database support
DATABASE_CONFIG = {
    'type': 'postgresql',
    'host': 'localhost',
    'port': 5432,
    'database': 'defendx',
    'user': 'defendx_user',
    'password': 'secure_password'
}
```

### SSL/HTTPS Setup

For production deployment with HTTPS:

```bash
# Generate SSL certificate
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365

# Run with SSL (future feature)
streamlit run src/app.py --server.sslCertFile cert.pem --server.sslKeyFile key.pem
```

## Maintenance

### Regular Maintenance Tasks

1. **Update Dependencies**:
   ```bash
   pip list --outdated
   pip install --upgrade -r requirements.txt
   ```

2. **Clean Data Directory**:
   ```bash
   # Remove old model files (>30 days)
   find data/models -name "*.pkl" -mtime +30 -delete
   ```

3. **Backup Data**:
   ```bash
   # Create backup
   tar -czf defendx-backup-$(date +%Y%m%d).tar.gz data/
   ```

### Monitoring

1. **Log Monitoring**:
   ```bash
   # Monitor Streamlit logs
   tail -f ~/.streamlit/logs/streamlit.log
   ```

2. **Performance Monitoring**:
   ```python
   # Add to application for monitoring
   import time
   import psutil
   
   start_time = time.time()
   memory_usage = psutil.virtual_memory().percent
   ```

This installation guide provides comprehensive instructions for setting up and maintaining the DefendX fraud detection system across different environments and use cases.
