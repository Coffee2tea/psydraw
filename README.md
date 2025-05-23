<p align="center">
  <img src="assets/logo2.png" alt="PsyDraw Logo" width="200"/>
</p>

<h1 align="center">PsyDraw: A Multi-Agent Multimodal System for Mental Health Screening in Left-Behind Children</h1>

<p align="center">
  <a href="README.md">
    <img src="https://img.shields.io/badge/Language-English-blue?style=for-the-badge" alt="English">
  </a>
  <a href="README_CN.md">
    <img src="https://img.shields.io/badge/语言-中文-blue?style=for-the-badge" alt="中文">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-GPL%203.0-green?style=for-the-badge" alt="License">
  </a>
  <a href="https://arxiv.org/abs/2412.14769">
    <img src="https://img.shields.io/badge/Paper-arXiv-red?style=for-the-badge" alt="Paper">
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/⚠️%20Professional%20Use%20Only-FF0000?style=for-the-badge" alt="Professional Use Only">
</p>

## ⚠️ Important Ethical Notice and Professional Use Guidelines

**CRITICAL: This system is strictly designed as a professional screening aid tool.**

This repository contains only the code structure of our project due to ethical considerations in mental health assessment. The system's prompts and analytical components are not open-sourced to prevent misuse and ensure proper application.

### Professional Use Only
- This tool is designed to assist qualified mental health professionals (psychiatrists, counselors, school advisors) in preliminary screenings. Non-professionals must not use it for self/peer assessments. All results require interpretation and validation by qualified professionals.
- The House-Tree-Person (HTP) test is not a diagnostic tool. Results are for reference only and cannot conclusively evaluate a participant's psychological state, personality, or capabilities.
- Participants' drawings and behaviors reflect subjective perspectives and transient emotional states, not long-term psychological traits. Results must not be used in contexts that may harm participants or influence evaluations, decisions, or selections
- Developers must protect participant confidentiality (personal data, artworks). Content will not be shared without consent, except in cases of force majeure (e.g., natural disasters, legal mandates).
- By using this tool, developers acknowledge its purpose, methodology, and limitations. Participants retain the right to withdraw at any time without penalty. The team assumes no liability for misuse or subsequent damages.

### Access to Full System
For research or clinical purposes requiring access to the complete system (including prompts), please contact us at: [project.htp@lyi.ai]. Access will only be granted after:
1. Verification of professional credentials
2. Review of intended use case
3. Instructions for use
4. Agreement to ethical guidelines and usage terms

### Risk Prevention
- Misuse of psychological assessment tools can lead to incorrect interpretations and potentially harmful outcomes
- The system should only be deployed in professional settings under qualified supervision
- All implementations must comply with relevant ethical guidelines and regulations in mental health assessment

## Project Overview
Left-behind children (LBCs), numbering over 66 million in China, face severe mental health challenges due to parental migration for work. Early screening and identification of at-risk LBCs is crucial, yet challenging due to the severe shortage of mental health professionals, especially in rural areas. While the House-Tree-Person (HTP) test shows higher child participation rates, its requirement for expert interpretation limits its application in resource-scarce regions. To address this challenge, we propose PsyDraw, a multi-agent system based on Multimodal Large Language Models that assists mental health professionals in analyzing HTP drawings. The system employs specialized agents for feature extraction and psychological interpretation, operating in two stages: comprehensive feature analysis and professional report generation. Evaluation of HTP drawings from 290 primary school students reveals that 71.03% of the analyzes achieved High Consistency with professional evaluations, 26.21% Moderate Consistency and only 2.41% Low Consistency. The system identified 31.03% of cases requiring professional attention, demonstrating its effectiveness as a preliminary screening tool. Currently deployed in pilot schools, PsyDraw shows promise in supporting mental health professionals, particularly in resource-limited areas, while maintaining high professional standards in psychological assessment.

<p align="center">
  <img src="assets/workflow.png" alt="PsyDraw Workflow"/>
  <br>
  <em>Figure 1: The workflow of PsyDraw</em>
</p>

## ✨ Key Features

<p align="center">
  <img src="https://img.shields.io/badge/HTP%20Analysis-Professional%20Grade-blue?style=for-the-badge" alt="HTP Analysis">
  <img src="https://img.shields.io/badge/Languages-EN%20%7C%20中文-blue?style=for-the-badge" alt="Languages">
  <img src="https://img.shields.io/badge/API-Professional%20Healthcare-blue?style=for-the-badge" alt="API">
  <img src="https://img.shields.io/badge/Web%20Tool-Supervised%20Assessment-blue?style=for-the-badge" alt="Web Tool">
</p>

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/LYiHub/psydraw.git
cd PsyDraw
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
- Copy `.env_example` file and rename to `.env`
- Fill in your API key and base URL

### Usage Methods

<p align="center">
  <img src="https://img.shields.io/badge/1-Direct%20Invocation-orange?style=for-the-badge" alt="Direct">
  <img src="https://img.shields.io/badge/2-API%20Integration-orange?style=for-the-badge" alt="API">
  <img src="https://img.shields.io/badge/3-Web%20Demo-orange?style=for-the-badge" alt="Web">
  <img src="https://img.shields.io/badge/4-Package%20App-orange?style=for-the-badge" alt="Package">
</p>

#### 1. Direct Invocation
```bash
bash run.sh
# or
python run.py --image_file example/example1.png --save_path example/example1_result.json --language en
```

#### 2. API Integration
```bash
python deploy.py --port 9557
```
Service runs on `http://127.0.0.1:9557`

#### 3. Web Demo
```bash
bash web_demo.sh
# or
streamlit run src/main.py
```

#### 4. Package Application
```bash
pyinstaller htp_analyzer.spec
```

## 📊 Case Studies
<p align="center">
  <img src="assets/case_study1.png" width="45%" />
  <img src="assets/case_study2.png" width="45%" /> 
</p>

## ⚖️ License

This project is licensed under the GPL-3.0 License. See the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

PsyDraw is strictly a professional screening aid tool. It must not be used as a standalone diagnostic tool or a substitute for professional medical evaluation. The system is designed to support, not replace, the expertise of qualified mental health professionals. Any implementation or use of this system must be under professional supervision.

## 📚 Citation

If you find this work helpful, please cite our paper:

```bibtex
@misc{zhang2024psydrawmultiagentmultimodalmental,
      title={PsyDraw: A Multi-Agent Multimodal System for Mental Health Screening in Left-Behind Children}, 
      author={Yiqun Zhang and Xiaocui Yang and Xiaobai Li and Siyuan Yu and Yi Luan and Shi Feng and Daling Wang and Yifei Zhang},
      year={2024},
      eprint={2412.14769},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.14769}, 
}
```

## 🌐 Streamlit Cloud Deployment

To deploy this application on Streamlit Cloud:

1. Push the repository to GitHub
2. Visit [Streamlit Cloud](https://streamlit.io/cloud) and sign in
3. Select "New app" and connect to your GitHub repository
4. Set the main file path to: `streamlit_app.py`
5. In the "Advanced settings" section:
   - Set Python version to 3.11 (recommended)
   - Add your OpenAI API key to the Secrets section with the key `OPENAI_API_KEY`

```toml
# Add this to the Secrets section in Streamlit Cloud
OPENAI_API_KEY = "your-openai-api-key"
```

6. Deploy the app and wait for the build process to complete

### Troubleshooting Deployment Issues

If you encounter errors during deployment:
- Check that your API key is correctly set in the Secrets section
- Verify that all dependencies are correctly specified in `requirements.txt`
- Examine the build logs for any specific error messages
