# Program title: Storytelling App

# Import part
import streamlit as st
from PIL import Image
from transformers import pipeline


# Function part
@st.cache_resource
def load_img2text_model():
    """Load the image captioning model once and reuse it after Streamlit reruns."""
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")


@st.cache_resource
def load_story_model():
    """Load the Hugging Face story generation model once."""
    return pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")


@st.cache_resource
def load_audio_generator():
    """Load the Hugging Face text-to-audio model once for better performance."""
    audio_generator = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
    return audio_generator


def img2text(image):
    """Create a short caption/scenario from the uploaded image."""
    image_to_text_model = load_img2text_model()
    text = image_to_text_model(image)[0]["generated_text"]
    return text


def trim_story(story, max_words=100):
    """Keep the story close to the assignment's 50-100 word target."""
    words = story.split()

    if len(words) <= max_words:
        return story.strip()

    trimmed_story = " ".join(words[:max_words]).strip()

    # Add a simple ending mark if the trim cuts the story in the middle.
    if trimmed_story[-1] not in ".!?":
        trimmed_story += "..."

    return trimmed_story


def make_story_short_enough(story, scenario):
    """Clean the generated story and keep it around 50-100 words."""
    story = " ".join(story.replace("\n", " ").split())

    # If the model gives a very short answer, add a simple ending so the final
    # result still feels like a complete short story.
    if len(story.split()) < 50:
        story += (
            f" Inspired by the scene of {scenario}, the moment became a gentle "
            "adventure. The character noticed small details, felt curious, and "
            "learned that even an ordinary picture can begin a memorable story."
        )

    return trim_story(story, max_words=100)


def text2story(scenario):
    """Generate a story from the image caption."""
    story_generator = load_story_model()
    story = story_generator(scenario)[0]["generated_text"]
    return make_story_short_enough(story, scenario)


def text2audio(story_text):
    """Convert the story text into audio using a Hugging Face TTS model."""
    audio_generator = load_audio_generator()
    speech_output = audio_generator(story_text)
    audio_array = speech_output["audio"]
    sample_rate = speech_output["sampling_rate"]
    return audio_array, sample_rate


# Main part
st.set_page_config(page_title="Your Image to Audio Story")
st.header("Turn Your Image to Audio Story")

uploaded_file = st.file_uploader(
    "Select an Image...",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is None:
    st.info("Please upload an image to begin.")
else:
    try:
        uploaded_image = Image.open(uploaded_file).convert("RGB")
    except Exception as error:
        st.error(f"Could not open this image file. Please try another image. Error: {error}")
        st.stop()

    # Section 1: Uploaded Image
    st.subheader("Uploaded Image")
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

    current_file_key = f"{uploaded_file.name}-{uploaded_file.size}"

    if st.session_state.get("file_key") != current_file_key:
        st.session_state["file_key"] = current_file_key
        st.session_state.pop("scenario", None)
        st.session_state.pop("story", None)
        st.session_state.pop("audio_array", None)
        st.session_state.pop("sample_rate", None)

    if st.button("Generate Story and Audio"):
        # Stage 1: Image to Text
        try:
            with st.spinner("Processing image captioning..."):
                scenario = img2text(uploaded_image)
        except Exception as error:
            st.error(f"Image captioning failed. Please try again. Error: {error}")
            st.stop()

        # Stage 2: Text to Story
        try:
            with st.spinner("Generating a short story..."):
                story = text2story(scenario)
        except Exception as error:
            st.error(f"Story generation failed. Please try again. Error: {error}")
            st.stop()

        # Stage 3: Story to Audio
        # The text-to-audio model returns audio data and a sample rate.
        try:
            with st.spinner("Generating audio data..."):
                audio_array, sample_rate = text2audio(story)
        except Exception as error:
            st.error(f"Audio generation failed. Please try again. Error: {error}")
            st.stop()

        st.session_state["scenario"] = scenario
        st.session_state["story"] = story
        st.session_state["audio_array"] = audio_array
        st.session_state["sample_rate"] = sample_rate

    if "scenario" in st.session_state:
        st.subheader("Generated Scenario / Caption")
        st.write(st.session_state["scenario"])

    if "story" in st.session_state:
        st.subheader("Generated Story")
        st.write(st.session_state["story"])

    if "audio_array" in st.session_state and "sample_rate" in st.session_state:
        st.subheader("Audio Player")
        if st.button("Play Audio"):
            st.audio(
                st.session_state["audio_array"],
                sample_rate=st.session_state["sample_rate"],
            )
