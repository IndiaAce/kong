# kong

# ThreatDetect-ML

## Overview

**ThreatDetect-ML** is a machine learning-driven tool designed to assist cybersecurity professionals in creating custom Splunk detections based on MITRE ATT&CK techniques and Splunk ESCU content. Leveraging `scikit-learn` and the `chatbot_ai` library, this application interacts with users to understand specific requirements, whether for threat detection, data model selection, or hunting for indicators of compromise (IOCs). The program outputs detection rules in YAML format, optimized for seamless deployment in Splunk environments.

## Features

- **Corpus Cloning**: Automatically clone the MITRE ATT&CK corpus and Splunk ESCU GitHub repository to build a comprehensive knowledge base for threat detection.
- **Machine Learning Training**: Train ML models on the cloned data to understand and predict detection patterns.
- **Interactive Chatbot Interface**: Use `chatbot_ai` to provide a conversational interface for defining detection requirements.
- **Detection Generation**: Generate Splunk detections (YAML files) based on user input, either by selecting TTPs (Tactics, Techniques, and Procedures) or by describing potential threats.
- **IOC-Based Detection**: Accept a list of IOCs (e.g., domains or IP addresses) and create Splunk queries to hunt for those indicators in your environment.

## Usage

ThreatDetect-ML is designed for scenarios where organizations need to:

- Automate the creation of custom detection rules based on established threat frameworks.
- Rapidly develop Splunk queries for threat hunting based on specific indicators.
- Engage with a conversational AI to simplify the detection development process for security professionals.

## Prerequisites

- **Python 3.10+**
- Libraries: `scikit-learn`, `chatbot_ai`, `pandas`, `requests`, and `PyYAML`
- Access to MITRE ATT&CK API or a local clone of the ATT&CK dataset
- **Git** for cloning the Splunk ESCU repository
