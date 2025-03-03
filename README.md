# GenAI CV - Job Description Matcher/Ranker

<center>
    <img src="./img/gen-ai-cv-jd-ranker.png" alt="GenAI CV - Job Description Matcher/Ranker" height="200"/>
</center>

GenAI CV - Job Description Matcher/Ranker is an application that uses Generative AI to evaluate and rank candidates based on how well their CVs match a given job description. This tool helps recruiters and hiring managers streamline the candidate selection process by providing a quantitative match score and a brief explanation for each candidate.

Link to the application: [https://genai-cv-job-disc-ranker.streamlit.app/](https://genai-cv-job-disc-ranker.streamlit.app/)

## How It Works

1. **Upload CVs**: Users can upload multiple CVs in PDF format.
2. **Enter Job Description**: Users provide a job description that outlines the requirements and qualifications for the position.
3. **Processing**: The application processes the CVs, converts the content into vectors using Hugging Face embeddings, and uses a language model to evaluate the match between the job description and each CV.
4. **Ranking**: The candidates are ranked based on the matching score, and a brief explanation is provided for each score.

## Technologies Used

- **Python**: The core programming language used for the application.
- **Streamlit**: For building the web interface.
- **LangChain**: For managing the language model and text processing.
- **Hugging Face Embeddings**: For converting CV content into vectors.
- **PyPDFLoader**: For loading and processing PDF documents.

## Potential Impact

This application can significantly improve the efficiency and effectiveness of the recruitment process by:

- Reducing the time and effort required to screen and evaluate candidates.
- Providing a standardized and objective way to assess candidate suitability.
- Enhancing the accuracy of candidate selection by leveraging advanced AI techniques.
- Enabling recruiters to focus on the most promising candidates, thereby improving the overall quality of hires.
