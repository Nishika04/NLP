from googletrans import Translator

def translate_text(text, target_language='en'):
    # Initialize the translator
    translator = Translator()

    # Translate the text to the target language
    translation = translator.translate(text, dest=target_language)

    return translation.text


#     # Return the translated text
#     return translation

# Step 1: Get user input in French
user_input_french = "Bonjour,certains de nos clients nous demandent à ce que leur numéro de TVA apparaisse sur les factures que nous leur envoyons. Pouvez-vous faire apparaitre ce numéro sur le rapport facture svp ? Comme fait sur la pièce jointe.Cordialement"  # Replace with the actual French input

# Step 2: Translate text to English
translated_input= translate_text(user_input_french, target_language='en')

# # Step 3: Analyze emotions in English
# emotion_results = analyze_emotions(translated_input)

# # Step 4: (Optional) Translate emotion results back to French
# translated_results, _ = translate_text(emotion_results, target_language='fr')

# Step 5: Display results
print(f"Original French Text: {user_input_french}")
print(f"Emotion Results: {translated_input}")