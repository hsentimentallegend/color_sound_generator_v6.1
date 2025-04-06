import streamlit as st
import numpy as np
from PIL import Image
import colorsys
from scipy.io import wavfile
import io
import base64
import urllib.parse

st.title("Color Sound Generator with Mode Switch")

# Mode selection
mode = st.radio("Select Mode", ("Random Mode", "Harmony Mode"))

# Image input
option = st.radio("Image Source", ("Camera", "Upload"))
if option == "Camera":
    img_file = st.camera_input("Take a Photo")
    if img_file is None:
        st.warning("Camera access denied. Please upload an image instead.")
else:
    img_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"], help="スマホではカメラを選択可能")

# セッション状態で画像とパラメータを保持
if 'img_array' not in st.session_state:
    st.session_state.img_array = None
    st.session_state.img_display = None
    st.session_state.wav_base64 = None
    st.session_state.mode_display = None
    st.session_state.vol_sine = 0.2
    st.session_state.vol_square = 0.2
    st.session_state.vol_saw = 0.2
    st.session_state.vol_noise = 0.2
    st.session_state.vol_gran = 0.2
    st.session_state.master_volume = 0.5
    st.session_state.bitcrush_enabled = False
    st.session_state.bit_depth = 4
    st.session_state.sample_rate_reduction = 8000
    st.session_state.refresh_key = 0  # 確実に初期化

if img_file is not None:
    img = Image.open(img_file).convert("RGB").resize((50, 50))
    img_display_raw = Image.open(img_file).convert("RGB")
    max_width, max_height = 720, 1280
    width, height = img_display_raw.size
    aspect_ratio = width / height
    if width > height:
        new_width = min(width, max_width)
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = min(height, max_height)
        new_width = int(new_height * aspect_ratio)
    img_display = img_display_raw.resize((new_width, new_height))
    st.image(img_display, caption="Selected Image", use_container_width=True)

    st.session_state.img_array = np.array(img)
    st.session_state.img_display = img_display

def generate_sound(img_array, mode, vol_sine, vol_square, vol_saw, vol_noise, vol_gran, master_volume, bitcrush_enabled, bit_depth, sample_rate_reduction):
    h, w, _ = img_array.shape
    third = h // 3
    part1 = img_array[:third, :, :]
    part2 = img_array[third:2*third, :, :]
    part3 = img_array[2*third:, :, :]

    rgb_part1 = np.mean(part1, axis=(0, 1))
    rgb_part2 = np.mean(part2, axis=(0, 1))
    rgb_part3 = np.mean(part3, axis=(0, 1))

    def rgb_to_hsv(rgb):
        r, g, b = rgb / 255.0
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        return h, s, v

    _, s1, v1 = rgb_to_hsv(rgb_part1)
    _, s2, v2 = rgb_to_hsv(rgb_part2)
    _, s3, v3 = rgb_to_hsv(rgb_part3)

    def value_to_freq(value, max_value=255):
        return 100 + (value / max_value) * 900

    fs = 44100
    duration = 10.0
    t = np.linspace(0, duration, int(fs * duration), False)

    def generate_oscillators(freq, t):
        sine = np.sin(2 * np.pi * freq * t)
        square = np.sign(np.sin(2 * np.pi * freq * t))
        sawtooth = 2 * (t * freq - np.floor(t * freq + 0.5))
        noise = np.random.uniform(-1, 1, len(t))
        granular = np.zeros_like(t)
        grain_size = int(fs * 0.05)
        for i in range(0, len(t) - grain_size, grain_size):
            if np.random.rand() < 0.3:
                grain = np.sin(2 * np.pi * freq * t[:grain_size])
                granular[i:i+grain_size] += grain
        return sine, square, sawtooth, noise, granular

    if mode == "Random Mode":
        freq_r1 = value_to_freq(rgb_part1[0])
        freq_v1 = value_to_freq(v1 * 255)
        freq_s1 = value_to_freq(s1 * 255)
        freq_r2 = value_to_freq(rgb_part2[0])
        freq_v2 = value_to_freq(v2 * 255)
        freq_s2 = value_to_freq(s3 * 255)
        freq_r3 = value_to_freq(rgb_part3[0])
        freq_v3 = value_to_freq(v3 * 255)
        freq_s3 = value_to_freq(s3 * 255)

        osc1_sine, osc1_square, osc1_saw, osc1_noise, osc1_gran = generate_oscillators(freq_r1, t)
        osc2_sine, osc2_square, osc2_saw, osc2_noise, osc2_gran = generate_oscillators(freq_v2, t)
        osc3_sine, osc3_square, osc3_saw, osc3_noise, osc3_gran = generate_oscillators(freq_s3, t)
        osc4_sine, osc4_square, osc4_saw, osc4_noise, osc4_gran = generate_oscillators(freq_r2, t)
        osc5_sine, osc5_square, osc5_saw, osc5_noise, osc5_gran = generate_oscillators(freq_v3, t)
        mode_display = "Random Mode"
    else:
        root_freq = value_to_freq(rgb_part1[0])
        avg_v = (v1 + v2 + v3) / 3
        if avg_v >= 0.5:
            chord_freqs = [
                root_freq,
                root_freq * (2 ** (4 / 12)),
                root_freq * (2 ** (7 / 12)),
                root_freq * (2 ** (11 / 12)),
                root_freq * (2 ** (14 / 12))
            ]
            mode_display = "Harmony Mode (Major 9th)"
        else:
            chord_freqs = [
                root_freq,
                root_freq * (2 ** (3 / 12)),
                root_freq * (2 ** (7 / 12)),
                root_freq * (2 ** (10 / 12)),
                root_freq * (2 ** (14 / 12))
            ]
            mode_display = "Harmony Mode (Minor 9th)"

        osc1_sine, osc1_square, osc1_saw, osc1_noise, osc1_gran = generate_oscillators(chord_freqs[0], t)
        osc2_sine, osc2_square, osc2_saw, osc2_noise, osc2_gran = generate_oscillators(chord_freqs[1], t)
        osc3_sine, osc3_square, osc3_saw, osc3_noise, osc3_gran = generate_oscillators(chord_freqs[2], t)
        osc4_sine, osc4_square, osc4_saw, osc4_noise, osc4_gran = generate_oscillators(chord_freqs[3], t)
        osc5_sine, osc5_square, osc5_saw, osc5_noise, osc5_gran = generate_oscillators(chord_freqs[4], t)

    base_sound = (
        vol_sine * (osc1_sine + osc2_sine + osc3_sine + osc4_sine + osc5_sine) +
        vol_square * (osc1_square + osc2_square + osc3_square + osc4_square + osc5_square) +
        vol_saw * (osc1_saw + osc2_saw + osc3_saw + osc4_saw + osc5_saw) +
        vol_noise * (osc1_noise + osc2_noise + osc3_noise + osc4_noise + osc5_noise) +
        vol_gran * (osc1_gran + osc2_gran + osc3_gran + osc4_gran + osc5_gran)
    )
    base_sound /= np.max(np.abs(base_sound)) if np.max(np.abs(base_sound)) > 0 else 1

    if bitcrush_enabled:
        levels = 2 ** bit_depth
        base_sound = np.round(base_sound * (levels - 1)) / (levels - 1)
        if sample_rate_reduction < fs:
            factor = fs // sample_rate_reduction
            base_sound = base_sound[::factor]
            base_sound = np.repeat(base_sound, factor)[:len(t)]

    mixed_sound = base_sound * master_volume

    wav_buffer = io.BytesIO()
    wavfile.write(wav_buffer, fs, (mixed_sound * 32767).astype(np.int16))
    wav_buffer.seek(0)
    wav_bytes = wav_buffer.getvalue()
    wav_base64 = base64.b64encode(wav_bytes).decode('utf-8')

    return wav_base64, mode_display

if st.session_state.img_array is not None:
    st.subheader("Sound Parameters")
    col1, col2 = st.columns(2)
    with col1:
        if mode == "Random Mode":
            st.session_state.vol_sine = st.slider("Sine Wave (R1)", 0.0, 1.0, st.session_state.vol_sine)
            st.session_state.vol_square = st.slider("Square Wave (V2)", 0.0, 1.0, st.session_state.vol_square)
            st.session_state.vol_saw = st.slider("Sawtooth Wave (S3)", 0.0, 1.0, st.session_state.vol_saw)
            st.session_state.vol_noise = st.slider("White Noise (R2)", 0.0, 1.0, st.session_state.vol_noise)
            st.session_state.vol_gran = st.slider("Granular (V3)", 0.0, 1.0, st.session_state.vol_gran)
        else:
            st.session_state.vol_sine = st.slider("Sine Wave (Root)", 0.0, 1.0, st.session_state.vol_sine)
            st.session_state.vol_square = st.slider("Square Wave (3rd)", 0.0, 1.0, st.session_state.vol_square)
            st.session_state.vol_saw = st.slider("Sawtooth Wave (5th)", 0.0, 1.0, st.session_state.vol_saw)
            st.session_state.vol_noise = st.slider("White Noise (7th)", 0.0, 1.0, st.session_state.vol_noise)
            st.session_state.vol_gran = st.slider("Granular (9th)", 0.0, 1.0, st.session_state.vol_gran)

    with col2:
        st.session_state.master_volume = st.slider("Master Volume", 0.0, 1.0, st.session_state.master_volume)
        st.session_state.bitcrush_enabled = st.checkbox("Enable Bitcrusher (Distortion)", value=st.session_state.bitcrush_enabled)
        if st.session_state.bitcrush_enabled:
            st.session_state.bit_depth = st.slider("Bit Depth", 1, 8, st.session_state.bit_depth)
            st.session_state.sample_rate_reduction = st.slider("Sample Rate Reduction", 1000, 44100, st.session_state.sample_rate_reduction)

    if st.button("Refresh Sound"):
        wav_base64, mode_display = generate_sound(
            st.session_state.img_array, mode, st.session_state.vol_sine, st.session_state.vol_square,
            st.session_state.vol_saw, st.session_state.vol_noise, st.session_state.vol_gran,
            st.session_state.master_volume, st.session_state.bitcrush_enabled,
            st.session_state.bit_depth, st.session_state.sample_rate_reduction
        )
        st.session_state.wav_base64 = wav_base64
        st.session_state.mode_display = mode_display
        st.session_state.refresh_key += 1
        st.write(f"Debug: refresh_key = {st.session_state.refresh_key}")  # デバッグ用

    if 'wav_base64' not in st.session_state or st.session_state.get('mode_display') != mode:
        wav_base64, mode_display = generate_sound(
            st.session_state.img_array, mode, st.session_state.vol_sine, st.session_state.vol_square,
            st.session_state.vol_saw, st.session_state.vol_noise, st.session_state.vol_gran,
            st.session_state.master_volume, st.session_state.bitcrush_enabled,
            st.session_state.bit_depth, st.session_state.sample_rate_reduction
        )
        st.session_state.wav_base64 = wav_base64
        st.session_state.mode_display = mode_display
        st.session_state.refresh_key += 1
        st.write(f"Debug: Initial refresh_key = {st.session_state.refresh_key}")  # デバッグ用
    else:
        wav_base64 = st.session_state.wav_base64
        mode_display = st.session_state.mode_display

    st.write(f"Generated Sound ({mode_display})")
    audio_html = f"""
    <audio controls>
        <source src="data:audio/wav;base64,{wav_base64}" type="audio/wav">
        Your browser does not support the audio element.
    </audio>
    """
    # keyをシンプルな文字列に
    audio_key = f"audio_{st.session_state.refresh_key}"
    st.write(f"Debug: Audio key = {audio_key}")  # デバッグ用
    st.markdown(audio_html, unsafe_allow_html=True, key=audio_key)

    tweet_text = f"Generated a unique sound from my #ColorCleanser artwork! Check it out: {mode_display}"
    tweet_url = f"https://twitter.com/intent/tweet?text={urllib.parse.quote(tweet_text)}"
    st.markdown(f'<a href="{tweet_url}" target="_blank"><button>Share on Twitter</button></a>', unsafe_allow_html=True)

    img_buffer = io.BytesIO()
    st.session_state.img_display.save(img_buffer, format="PNG")
    img_buffer.seek(0)
    st.download_button(
        label="Download Image",
        data=img_buffer,
        file_name="color_image.png",
        mime="image/png"
    )

st.sidebar.markdown("---")
st.sidebar.write("Created by Hiroshi Mehata")
st.sidebar.write("Extension app for Color Cleanser Exhibition")
st.sidebar.markdown("[Website](https://www.mehatasentimentallegend.com/the-story-of-color-cleanser)")
st.sidebar.image("c-001.jpg", caption="Color Cleanser", use_container_width=True)
