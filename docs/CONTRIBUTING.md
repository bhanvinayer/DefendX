# Contributing to DefendX

We welcome contributions to the DefendX Multi-Agent Fraud Detection System! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Development Standards](#development-standards)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of background, experience level, or identity.

### Standards

**Positive behavior includes:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behavior includes:**
- Harassment, trolling, or discriminatory comments
- Personal attacks or political arguments
- Publishing private information without permission
- Any conduct that could be considered inappropriate in a professional setting

## Getting Started

### Prerequisites

Before contributing, ensure you have:
- Python 3.8 or higher
- Git installed and configured
- Familiarity with the project architecture (see `docs/ARCHITECTURE.md`)
- Understanding of the technology stack (see `docs/TECH_STACK.md`)

### Areas for Contribution

We welcome contributions in these areas:

**🔒 Security & Fraud Detection**
- Enhanced anomaly detection algorithms
- New behavioral feature extraction methods
- Security vulnerability assessments
- Performance optimization

**🎨 User Interface & Experience**
- UI/UX improvements
- Accessibility enhancements
- New theme options
- Mobile responsiveness

**📊 Analytics & Visualization**
- Advanced analytics dashboards
- New visualization types
- Real-time monitoring features
- Reporting capabilities

**🧪 Testing & Quality Assurance**
- Unit test coverage improvement
- Integration testing
- Performance testing
- Security testing

**📚 Documentation**
- Code documentation
- User guides and tutorials
- API documentation
- Installation guides

**🚀 Performance & Scalability**
- Algorithm optimization
- Memory usage improvements
- Database integration
- Multi-user support

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/defendx.git
cd defendx

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/defendx.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv defendx-dev
source defendx-dev/bin/activate  # On Windows: defendx-dev\Scripts\activate

# Install development dependencies
pip install -r requirements.txt

# Verify setup
streamlit run src/app.py
```

### 3. Project Structure

```
DefendX/
├── src/
│   └── app.py              # Main application (2743 lines)
├── docs/
│   ├── README.md           # Project overview
│   ├── ARCHITECTURE.md     # System architecture
│   ├── API.md             # API reference
│   ├── INSTALLATION.md    # Installation guide
│   ├── USER_GUIDE.md      # User documentation
│   ├── TECH_STACK.md      # Technology stack
│   └── CONTRIBUTING.md    # This file
├── data/
│   ├── models/            # ML models
│   ├── session_log.csv    # Session data
│   └── sample_data/       # Test data
├── requirements.txt       # Dependencies
└── README.md             # Main project README
```

## Contributing Guidelines

### Branch Naming

Use descriptive branch names that follow this pattern:
- `feature/description` - For new features
- `bugfix/description` - For bug fixes
- `security/description` - For security improvements
- `docs/description` - For documentation updates
- `refactor/description` - For code refactoring

### Commit Messages

Write clear, descriptive commit messages:

```
feat: add real-time anomaly detection alerts

- Implement real-time alert system for fraud detection
- Add notification components to UI
- Include configurable alert thresholds
- Update documentation for new feature

Closes #123
```

**Commit message format:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `security:` - Security improvements

### Code Style

**Python Code Style:**
- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Include type hints where possible
- Add docstrings for all public functions and classes

**Example:**
```python
def extract_behavioral_features(
    keystroke_data: List[Dict], 
    reference_text: str
) -> Dict[str, float]:
    """
    Extract behavioral features from keystroke timing data.
    
    Args:
        keystroke_data: List of keystroke events with timestamps
        reference_text: The reference text being typed
        
    Returns:
        Dictionary containing extracted behavioral features
        
    Raises:
        ValueError: If keystroke_data is empty or invalid
    """
    # Implementation here
    pass
```

## Pull Request Process

### 1. Prepare Your Contribution

```bash
# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name

# Make your changes
# ... develop your feature ...

# Test your changes
streamlit run src/app.py

# Commit your changes
git add .
git commit -m "feat: descriptive commit message"
```

### 2. Submit Pull Request

1. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request:**
   - Go to GitHub and create a pull request
   - Use the pull request template
   - Provide clear description of changes
   - Link any related issues

3. **PR Description Template:**
   ```markdown
   ## Description
   Brief description of the changes made.
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Security improvement
   - [ ] Performance optimization
   
   ## Testing
   - [ ] Tested locally
   - [ ] Added unit tests
   - [ ] Updated documentation
   
   ## Screenshots (if applicable)
   Add screenshots showing UI changes.
   
   ## Related Issues
   Closes #issue_number
   ```

## Issue Reporting

### Bug Reports

Use the bug report template:

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '...'
3. See error

**Expected behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment:**
- OS: [e.g., Windows 10, macOS, Ubuntu]
- Python version: [e.g., 3.9.7]
- Browser: [e.g., Chrome 96]

**Additional context**
Any other context about the problem.
```

### Feature Requests

Use the feature request template:

```markdown
**Is your feature request related to a problem?**
A clear description of the problem.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Alternative solutions or features you've considered.

**Additional context**
Any other context, mockups, or examples.
```

### Security Issues

**For security vulnerabilities:**
- Do NOT create public issues
- Email security concerns to the maintainers
- Include detailed description and reproduction steps
- Allow time for assessment and fix before disclosure

## Development Standards

### Code Quality

**Required standards:**
- All functions must have docstrings
- Type hints for function parameters and returns
- Error handling for edge cases
- Follow existing code patterns and conventions

### Multi-Agent Architecture

When contributing to the agent system, maintain:
- **KeystrokeAgent**: Keystroke capture and analysis
- **BehaviorModelAgent**: ML model training and prediction
- **FraudDetectionAgent**: Threat analysis and alerting
- **DataManagerAgent**: Data persistence and management

### Security Standards

**Security considerations:**
- No hardcoded credentials or secrets
- Input validation for all user inputs
- Secure file handling practices
- Privacy protection for user data
- Local processing only (no external APIs)

## Current Architecture

The DefendX system is implemented as a single Streamlit application (`src/app.py`) with:
- **2743 lines** of integrated code
- **Four agent classes** for specialized functionality
- **Theme system** with light/dark modes and accessibility features
- **Professional UI** with multiple pages and real-time analytics
- **Local data storage** using CSV and JSON formats

## Getting Help

### Communication Channels

**For development questions:**
- Create GitHub discussions for general questions
- Use issues for specific problems
- Join community chat (if available)

### Resources

**Helpful documentation:**
- `docs/ARCHITECTURE.md` - System architecture
- `docs/API.md` - API reference
- `docs/TECH_STACK.md` - Technology overview
- `docs/USER_GUIDE.md` - User documentation
- `docs/INSTALLATION.md` - Setup instructions

## Recognition

### Contributors

We recognize contributions in several ways:
- Contributors list in README
- Release notes acknowledgment
- Community highlights
- Maintainer invitation for significant contributors

### Types of Contributions Valued

- Code contributions (features, fixes, optimizations)
- Documentation improvements
- Bug reports and testing
- Community support and mentoring
- Design and UX improvements
- Security audits and improvements

Thank you for contributing to DefendX! Your efforts help make fraud detection more accessible and effective for everyone.
