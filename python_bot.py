import os.path
import subprocess

import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes,filters
from pydub import AudioSegment

from sign_writing_approach.model.sign_writing_model import SignWritingModel
from spoken_to_text.spoken_to_text import AudioRecorder
from flow_manager import FlowManager

TOKEN = '5993731273:AAHnf_LvkKJO-oWdXf_O20eDvRXKBF7Bkco'
FFMPEG_PATH = 'C:\\projects\\ffmpeg\\bin'
os.environ["ffmpeg"] = FFMPEG_PATH

AudioSegment.converter = FFMPEG_PATH
AudioSegment.ffmpeg = FFMPEG_PATH

flow_manager = FlowManager(model=SignWritingModel(
    checkpoint_path='./sign_writing_approach/model/sw-model-v1-all-data.ckpt'),
    encoded_vectors_path='./sign_writing_approach/store_vectors/signsuisse-Vectors-all-model.jsonl')

# TODO: fix audio shit, create api for the model(easy), queues ?
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}!",
    )


async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    file_name = 'audio'
    voice = update.message.voice
    file = await voice.get_file()

    response = requests.get(file.file_path)

    with open(f'{file_name}.oga', 'wb') as f:
        f.write(response.content)

    if os.path.exists(f'{file_name}.wav'):
        os.remove(f'{file_name}.wav')

    subprocess.run(['ffmpeg', '-i', f'{file_name}.oga', f'{file_name}.wav'])

    recorder = AudioRecorder()
    text = recorder.convert_to_text(wav_filename=f'{file_name}.wav')
    await update.message.reply_text(text)

    flow_manager.run(text)

    if not text.__contains__('Exception'):
        await context.bot.send_video(chat_id=update.effective_chat.id,
                                     video=open('s2m0a2a63bda69676afdc0a968924754578.mp4', 'rb'))
    os.remove(f'{file_name}.oga')
    os.remove(f'{file_name}.wav')


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Echo the user message."""
    txt = update.message.text
    await update.message.reply_text(txt)


def main():
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("text", echo))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice_message))

    print("Running application...")
    # application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))
    application.run_polling()


if __name__ == "__main__":
    main()