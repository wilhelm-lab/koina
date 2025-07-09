# Contributor Guidelines

Welcome! We're thrilled you're interested in contributing to Koina. These guidelines are designed to help you understand how you can contribute effectively and ensure a smooth collaboration.

---

## Getting Started
- **Familiarize Yourself**: Make sure to checkout the main README, [publication](https://www.biorxiv.org/content/10.1101/2024.06.01.596953v1.full) and [website](https://koina.wilhelmlab.org/).  
- **Check the Issue Tracker**: Look for open [issues](https://github.com/wilhelm-lab/koina/issues) labeled “good first issue” or “help wanted” to find tasks suited for beginners. 
- **Ask Questions**: If you're unsure about anything, don't hesitate to contact us.  

## Contribution Types
We welcome contributions in various forms, with a special emphasis on:  
- **Documentation Models**: Improving documentation to help users better understand and evaluate models.  
  - **Documentation Questionnaire**: We've created a documentation [questionnaire](docs/DOME.md) based on the DOME documentation guidelines. Following this guide for writing the documentation of a contributed model is encouraged.  
  - **Clarity and Detail**: Ensure documentation is clear, concise, and provides all necessary information for users to understand the model's purpose, limitations, and usage.  
- **New Models**: Contributing new machine learning (ML) models to enhance accessibility and impact. Check out the [this guide](docs/README.md) to get started!

Other contribution types include:  
- **Code**: Bug fixes, feature implementations, or optimizations.  
- **Design**: UI/UX improvements, graphics, or branding.  
- **Testing**: Reporting bugs, writing test cases, or improving test coverage.  
- **Ideas**: Suggesting new features or improvements.  

## Quality Control
We don't evaluate models made available on Koina based on their prediction accuracy/performance. We invite all authors of ML models to contribute their models to Koina to improve their impact and accessibility.  

## Validation
We heavily encourage you to provide test data for their models.  
- **Test Data**: Providing test data ensures that predictions are not changing due to unexpected effects of dependencies, infrastructure or optimizations.  
- **Testing Schema**: We've set up a [testing schema](clients/python/test/lib.py) to ensure the long-term maintainability of models. 

## Versioning
Older versions of models should stay available to enable users to reproduce past results.
- **Non-Prediction-Affecting Updates**: Updates that don't affect the generated predictions, such as performance improvements in pre- and post-processing scripts, don't require a separate version of the model.  
- **Versioning Mechanisms**:  
  - **Primary Method**: Use numbered folders in the model directory. This ensures all versions of a model are freely available via Koina. This method is limited to changes that don't require the config file to be adjusted.  
  - **Secondary Method**: For changes requiring adjustments to the config file, create a separate version of the model with an incrementing numbered suffix (`<model name>_<version number>`).  

## Contribution Process
1. **Fork the Repository**: Create a fork of the project repository.  
2. **Create a Branch**: Use a descriptive branch name (e.g., `feature/add-login` or `fix/bug-123`).  
3. **Make Your Changes**: Follow the project's coding standards and conventions.  
4. **Test Your Changes**: Ensure your changes work as expected and don't introduce new issues.  
5. **Submit a Pull Request (PR)**:  
   - Provide a clear and concise description of your changes.  
   - Reference related issues (e.g., “Closes #123”).  
   - Be responsive to feedback during the review process.  

## Code Style and Standards
- Follow the project's coding conventions (e.g., indentation, naming, etc.).  
- Include comments where necessary to explain complex logic.  
- Ensure your code is linted and formatted according to the project's guidelines.  

## Reporting Issues
When reporting bugs or suggesting improvements:
- Open issues on [GitHub](https://github.com/wilhelm-lab/koina/issues)
- Include steps to reproduce the issue, expected behavior, and actual behavior.  
- Provide details about your environment (e.g., OS, browser, version).  

## Communication
- Be respectful and inclusive in all interactions.   
- Keep discussions focused and constructive.  

## Licensing
By contributing, you agree to license your work under the project's open-source license (Apache 2.0).  

## Thank you!  
Thank you for your interest in contributing to Koina! Your efforts help make this project better for everyone.  

2025-03-18
