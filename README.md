# Specialized Agent Chatbot

## Overview

This project features a chatbot that takes user prompts and automatically detects the most suitable agent to handle the request. It assigns the task to the chosen agent and initiates a chat session with the user. The system leverages natural language processing (NLP) and machine learning models to match user queries with specialized agents.

## Features

- **User Prompt Handling**: Accepts user input and processes it to determine the appropriate agent.
- **Agent Detection**: Identifies and recommends the best-suited agent based on the user's prompt using NLP and embedding similarity.
- **Task Assignment**: Automatically assigns the task to the selected agent.
- **Chatbot Interaction**: Initiates a conversation with the user to handle the task.

## Installation

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    ```

2. **Navigate to the Project Directory:**

    ```bash
    cd your-repo-name
    ```

3. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Set Up Environment Variables:**

    Create a `.env` file in the root directory and add your OpenAI API key:

    ```plaintext
    OPENAI_API_KEY=your_openai_api_key
    ```

## Usage

1. **Run the Application:**

    ```bash
    streamlit run app.py
    ```

2. **Interact with the Chatbot:**

    - Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).
    - Enter a user prompt in the sidebar and click "Save Prompt" to start the interaction.
    - If multiple agents are suitable, select the preferred one from the dropdown menu.

3. **Chat Interface:**

    - Once a prompt is saved and an agent is selected, the chat interface will display the conversation.
    - You can ask further questions or start a new conversation using the sidebar buttons.

## Project Structure

- **`app.py`**: Main application file containing the chatbot logic.
- **`requirements.txt`**: List of required Python packages.
- **`.env`**: Environment file for storing sensitive information such as API keys.

## Contributing

1. **Fork the Repository** and create a new branch:

    ```bash
    git checkout -b feature/your-feature
    ```

2. **Make Your Changes** and commit them:

    ```bash
    git commit -am 'Add new feature'
    ```

3. **Push to Your Branch**:

    ```bash
    git push origin feature/your-feature
    ```

4. **Create a Pull Request** on GitHub.


## Acknowledgments

- **OpenAI** for providing the language model API.
- **spaCy** for natural language processing tools.
- **Sentence Transformers** for embedding and similarity calculations.

## Contact

For any questions or inquiries, please reach out to me at [sachinparmar1128@gmail.com](mailto:sachinparmar1128@gmail.com).

---

Feel free to adjust any details to better fit your specific implementation or preferences.
