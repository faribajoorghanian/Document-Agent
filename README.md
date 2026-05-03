# Multimodal Document AI Agent 📄🔍

An end-to-end, stateful AI agent designed to bridge Computer Vision and Natural Language Processing. Built with **LangGraph**, this production-grade pipeline autonomously processes raw, noisy document images, extracts structural layouts, reads text, and applies LLM reasoning to answer user queries accurately.

## 🚀 Key Features

*   **Vision Preprocessing:** Utilizes **OpenCV** (Adaptive Thresholding, Grayscale conversion) to clean document images and mitigate real-world physical anomalies like shadows and noise.
*   **Intelligent Layout Analysis:** Employs **YOLOv8** to detect and isolate structural document regions. It implements a custom spatial sorting algorithm to arrange visual bounding boxes into a semantic reading order.
*   **OCR Extraction:** Digitizes text from the cropped visual regions using EasyOCR, effectively converting unstructured visual data into structured textual context.
*   **LLM Reasoning:** Integrates a pre-trained RoBERTa model via Hugging Face `transformers` to perform context-aware Question Answering (QA).
*   **Agentic Orchestration:** Manages the entire multi-step pipeline using **LangGraph**, ensuring a modular, highly scalable, and stateful architecture suitable for production environments.

## 🧠 System Architecture

```mermaid
graph TD
    %% Inputs
    I[Document Image] --> Preprocess
    Q[User Query] --> Reasoning

    %% Computer Vision Pipeline
    subgraph Vision Pipeline [Computer Vision Pipeline]
        Preprocess[Preprocess Node<br/>OpenCV: Binarization]
        Layout[Layout Analysis Node<br/>YOLOv8: Region Detection]
        Preprocess --> Layout
    end

    %% NLP & OCR Pipeline
    subgraph Language Pipeline [NLP & Reasoning Pipeline]
        OCR[OCR Extraction Node<br/>EasyOCR: Text Reading]
        Reasoning[LLM Reasoning Node<br/>RoBERTa: Question Answering]
    end

    %% Edge Connections
    Layout -->|Cropped Regions| OCR
    OCR -->|Extracted Text Context| Reasoning
    
    %% Output
    Reasoning --> Output((Final Answer))

    %% Styling
    classDef vision fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef nlp fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px;
    classDef input fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    
    class Preprocess,Layout vision;
    class OCR,Reasoning nlp;
    class I,Q input;
```

## 🛠️ Tech Stack

*   **Computer Vision:** OpenCV, Ultralytics (YOLO)
*   **NLP & OCR:** Hugging Face, EasyOCR
*   **Orchestration:** LangGraph
*   **Core:** Python, PyTorch

## 💻 Quick Start

### Prerequisites
```bash
pip install langgraph langchain easyocr ultralytics opencv-python transformers
