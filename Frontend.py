import streamlit as st
import pickle

# Load the pre-trained model and vectorizer
model = pickle.load(open('spam123.pkl', 'rb'))
cv = pickle.load(open('vec123.pkl', 'rb'))

# Function to apply custom CSS styles
def apply_custom_css():
    st.markdown(
        """
        <style>
        body {
            background-color: #121212;  /* Dark background */
            font-family: 'Arial', sans-serif;
            color: black;  /* Default text color set to black */
        }
        .title {
            font-size: 3em;
            color: #28A745;
            text-align: center;
            font-weight: bold;
            margin-top: 40px;
        }
        .subheader {
            font-size: 1.5em;
            color: #28A745;
            text-align: center;
            margin-top: 20px;
        }
        .text-area {
            border-radius: 8px;
            width: 100%;
            padding: 10px;
            font-size: 1em;
            margin-top: 20px;
            height: 200px;
            border: 2px solid #FFCC00;  /* Border color */
            background-color: #2C2C2C;  /* Darker background for text area */
            color: black;  /* Text inside the text area will be black */
        }
        .text-area:focus {
            border-color: #FFCC00;  /* Focus border color */
            outline: none;
        }
        .button {
            background-color: #FFCC00;
            color: black;
            border: none;
            padding: 12px 20px;
            text-align: center;
            font-size: 1em;
            cursor: pointer;
            border-radius: 5px;
            margin-top: 20px;
        }
        .button:hover {
            background-color: #FF9900;
        }
        .result {
            font-size: 1.2em;
            font-weight: bold;
            margin-top: 30px;
            color: green;  /* Make the result text green */
        }
        .footer {
            text-align: center;
            font-size: 0.8em;
            color: #888;
            margin-top: 50px;
        }
        </style>
        """, unsafe_allow_html=True
    )

def main():
    apply_custom_css()  # Apply custom styles

    # Title and description
    st.markdown('<p class="title">Email Spam Classification</p>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">This is a Machine Learning application to classify spam and ham emails.</p>', unsafe_allow_html=True)

    # Input field for email text
    user_input = st.text_area(
        "Enter an Email to classify",
        height=150,
        key="email_input",
        help="Type the email text here."
    )

    # Button to trigger classification
    if st.button("Classify", key="classify_button", help="Click to classify the email as spam or ham"):
        if user_input:
            data = [user_input]
            vec = cv.transform(data).toarray()
            result = model.predict(vec)
            
            if result[0] == 0:
                st.success('This is a Ham message!', icon="✅")
            else:
                st.error("This is a Spam email", icon="❌")
        else:
            st.warning("Please enter an email to classify", icon="❗")
    
    # Footer information
    st.markdown('<p class="footer">Made with ❤️ by R Bhuvan Dhanush</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
