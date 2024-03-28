import streamlit as st
import tensorflow as tf
import tensorflow_text as text


# Define the function for printing examples
def print_example(input_text, score):
    st.write(f"Input: {input_text}")
    st.write(f"Score: {score:.4f}")
    st.write()

# Load my model
reloaded_model = tf.keras.models.load_model('./imdb_bert')

# Define the Streamlit app
def main():
    st.title("Classify Text: Sentiment Analysis")
    user_input = st.text_input("Enter your review here:")
    predict_button = st.button("Predict")

    if predict_button:
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            # Perform prediction
            result = tf.sigmoid(reloaded_model(tf.constant([user_input])))
            print_example(user_input, result.numpy()[0][0])

if __name__ == "__main__":
    main()
