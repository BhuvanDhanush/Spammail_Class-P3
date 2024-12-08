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
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
        
        body {
            background: linear-gradient(45deg, #121212, #1e1e1e);
            font-family: 'Poppins', sans-serif;
            color: #E0E0E0;
            margin: 0;
            padding: 0;
        }
        
        .title {
            font-size: 3.5em;
            color: #FFCC00;
            text-align: center;
            font-weight: 600;
            margin-top: 40px;
            animation: fadeIn 2s ease-in-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        .subheader {
            font-size: 1.5em;
            color: #FFCC00;
            text-align: center;
            margin-top: 20px;
        }

        .text-area {
            border-radius: 8px;
            width: 100%;
            padding: 12px;
            font-size: 1.1em;
            margin-top: 30px;
            height: 200px;
            border: 2px solid #333;
            background-color: #2C2C2C;
            color: #E0E0E0;
            transition: 0.3s;
        }

        .text-area:focus {
            border-color: #FFCC00;
            outline: none;
            background-color: #3A3A3A;
        }

        .button {
            background-color: #FFCC00;
            color: black;
            border: none;
            padding: 14px 24px;
            text-align: center;
            font-size: 1.1em;
            cursor: pointer;
            border-radius: 10px;
            margin-top: 30px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
            transition: 0.3s ease;
        }

        .button:hover {
            background-color: #FF9900;
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }

        .result {
            font-size: 1.4em;
            font-weight: 600;
            margin-top: 30px;
            text-align: center;
            transition: 0.3s ease;
        }

        .footer {
            text-align: center;
            font-size: 0.8em;
            color: #888;
            margin-top: 60px;
        }

        .loading {
            display: block;
            margin: 40px auto;
            text-align: center;
            font-size: 1.5em;
            color: #FFCC00;
        }
        
        </style>
        """, unsafe_allow_html=True
    )

def main():
    apply_custom_css()  # Apply custom styles

    # Title and description
    st.markdown('<p class="title">Welcome to the Spam Classifier!</p>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">I can help you classify your emails into "Spam" or "Ham". Just paste the email below!</p>', unsafe_allow_html=True)

    # Input field for email text
    user_input = st.text_area("Paste the email you want to classify:", height=150, key="email_input", help="Simply paste your email text here.", max_chars=1000)
    
    # Button to trigger classification
    if st.button("Classify my Email!", key="classify_button", help="Click here to classify your email as either Spam or Ham"):
        if user_input:
            # Show loading indicator while processing
            with st.spinner("Hang tight! I'm processing your email..."):
                data = [user_input]
                vec = cv.transform(data).toarray()
                result = model.predict(vec)
                
                # Display the result with a transition
                if result[0] == 0:
                    st.success('🎉 This email is **Ham**! You’re safe. No spam here!', icon="✅")
                else:
                    st.error("🚨 Oops, this email is **Spam**! Be cautious.", icon="⚠️")
        else:
            st.warning("❗ Hey there! Please paste an email first so I can help you classify it.", icon="❗")

    # Footer information
    st.markdown('<p class="footer">Made with ❤️ by Your Friendly Spam Classifier Bot</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
