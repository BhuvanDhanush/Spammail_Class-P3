import streamlit as st
import pickle

# Load the pre-trained model and vectorizer
model = pickle.load(open('spam123.pkl', 'rb'))
cv = pickle.load(open('vec123.pkl', 'rb'))

# Function to apply custom CSS styles with shadow effects
def apply_custom_css():
    st.markdown(
        """
        <style>
        body {
            background-color: #FFFFFF;
            font-family: 'Arial', sans-serif;
            color: #FFFFFF;  /* Black text */
        }
        .title {
            font-size: 3em;
            color: #000000;  /* Green text */
            text-align: center;
            font-weight: bold;
            margin-top: 40px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);  /* Shadow effect for title */
        }
        .subheader {
            font-size: 1.5em;
            color: #FF0000;  /* Green text */
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
            border: 2px solid #28A745;  /* Green border */
            background-color: #2C2C2C;  /* Darker background for text area */
            color: #000000;  /* Black text */
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);  /* Shadow effect for text area */
        }
        .text-area:focus {
            border-color: #FF0000;  /* Green focus border */
            outline: none;
        }
        .button {
            background-color: #28A745;  /* Green button */
            color: black;
            border: none;
            padding: 12px 20px;
            text-align: center;
            font-size: 1em;
            cursor: pointer;
            border-radius: 5px;
            margin-top: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);  /* Shadow effect for button */
        }
        .button:hover {
            background-color: #218838;  /* Darker green on hover */
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);  /* Darker shadow on hover */
        }
        .result {
            font-size: 1.2em;
            font-weight: bold;
            margin-top: 30px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);  /* Shadow effect for result */
            padding: 10px;
            background-color: #1F1F1F;  /* Dark background for result */
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
    st.markdown('<p class="footer">Made by R Bhuvan Dhanush</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
