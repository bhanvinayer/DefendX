# Contributing to DefendX - Behavior-Based Fraud Detection

## 🚀 Getting Started

Thank you for your interest in contributing to the DefendX Behavior-Based Fraud Detection system! This project implements a privacy-first, on-device multi-agent system for detecting fraud through keystroke dynamics and behavioral biometrics.

## 🔧 Development Setup

### Prerequisites
- Python 3.8 or higher
- Git

### Installation
```bash
# Clone the repository
git clone https://github.com/bhanvinayer/DefendX.git
cd DefendX

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## 🏗️ Project Structure
```
DefendX/
├── app.py                    # Main Streamlit application
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
├── CONTRIBUTING.md          # This file
├── LICENSE                  # Project license
├── data/                    # Data storage directory
│   ├── models/             # Trained ML models
│   └── sample_data/        # Sample datasets
├── docs/                   # Additional documentation
├── tests/                  # Unit tests
└── assets/                 # Images, diagrams, etc.
```

## 🤖 Multi-Agent Architecture

### Agents Overview
1. **KeystrokeAgent**: Captures and analyzes keystroke patterns
2. **BehaviorModelAgent**: Trains and maintains user behavior models
3. **FraudDetectionAgent**: Detects anomalies and fraud patterns
4. **DataManagerAgent**: Handles data persistence and management

## 🔍 Code Style Guidelines

### Python Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to all classes and methods
- Keep functions focused and small
- Use meaningful variable names

### Example Code Structure
```python
class ExampleAgent:
    """Brief description of the agent's purpose"""
    
    def __init__(self):
        """Initialize agent with required parameters"""
        pass
    
    def process_data(self, data: Dict) -> Dict:
        """
        Process input data and return results
        
        Args:
            data: Input data dictionary
            
        Returns:
            Processed results dictionary
        """
        pass
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_keystroke_agent.py

# Run with coverage
python -m pytest tests/ --cov=.
```

### Writing Tests
- Write tests for all new functionality
- Aim for 80%+ code coverage
- Use descriptive test names
- Test both success and failure cases

## 📝 Commit Guidelines

### Commit Message Format
```
<type>(<scope>): <description>

<body>

<footer>
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

### Examples
```
feat(keystroke): add enhanced timing metrics

Add hold time, gap time, and flight time tracking to improve
behavioral analysis accuracy.

Closes #123
```

## 🐛 Issue Reporting

### Bug Reports
When reporting bugs, please include:
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details (OS, Python version)
- Screenshots if applicable

### Feature Requests
For new features, please provide:
- Clear description of the feature
- Use case and motivation
- Proposed implementation approach
- Any relevant examples

## 🔒 Security Considerations

### Privacy Requirements
- All processing must remain on-device
- No external API calls without explicit permission
- User data must be handled securely
- Follow privacy-by-design principles

### Security Best Practices
- Validate all input data
- Use secure coding practices
- Handle errors gracefully
- Log security-relevant events

## 🎯 Pull Request Process

### Before Submitting
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Update documentation if needed
7. Commit your changes
8. Push to your fork
9. Create a Pull Request

### PR Requirements
- [ ] Code follows style guidelines
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] Commit messages are descriptive
- [ ] No merge conflicts

### Review Process
1. Automated checks must pass
2. Code review by maintainers
3. Address feedback if any
4. Final approval and merge

## 📚 Documentation

### Adding Documentation
- Update README.md for major changes
- Add inline comments for complex logic
- Include docstrings for all public methods
- Provide examples for new features

### Documentation Style
- Use clear, concise language
- Include code examples
- Add screenshots for UI changes
- Keep documentation up-to-date

## 🌟 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes for significant contributions
- Project acknowledgments

## 📞 Getting Help

### Communication Channels
- GitHub Issues: For bug reports and feature requests
- GitHub Discussions: For questions and general discussion
- Email: Direct contact with maintainers

### Resources
- [Project Documentation](README.md)
- [API Reference](docs/api-reference.md)
- [Architecture Guide](docs/architecture.md)

## 📄 License

By contributing to DefendX, you agree that your contributions will be licensed under the same license as the project.

---

Thank you for contributing to DefendX! Together, we're building better security through behavioral biometrics. 🚀
