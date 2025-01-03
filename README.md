## Architecture Diagram, Reference - Alejandro
![image](https://github.com/user-attachments/assets/987d46b5-b4ea-456b-9003-153848ec5bc6)
## Overview
This project implements the multimodal RAG using GPT-4 and Unstructed.io. It processes the pdf files to extract tables, images and text using Unstructerd.io library, summarize it then embed the summaries along with the original input and insert into Vector Database. The system is built using `Streamlit` for the user interface and answers the users question using RAG mechanism.

To run the application install the requirements and command -
```
streamlit run .\app.py
```

**Results-**

![image](https://github.com/user-attachments/assets/c57917ed-9f28-4691-afa2-0e0fe92e2d50)

**Image query-** 

![image](https://github.com/user-attachments/assets/8a36c0e8-56ef-479d-965d-fbf0a196ceee)
![image](https://github.com/user-attachments/assets/43de7e1a-3d6f-49d6-a174-e5963178fef5)



![image](https://github.com/user-attachments/assets/a35690d6-993d-404a-8b70-16ea712d4cda)
