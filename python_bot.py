import io
import os.path

import telegram
import logging
from telegram import Update, ForceReply
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes,filters
from io import BytesIO

from spoken_to_text import spoken_to_text

FFMPEG_PATH= 'C:\\Users\\tommy\\Downloads\\ffmpeg-master-latest-win64-gpl\\ffmpeg-master-latest-win64-gpl\\bin'
os.environ["FFMPEG_PATH"] = FFMPEG_PATH
from pydub import AudioSegment

AudioSegment.converter = FFMPEG_PATH
AudioSegment.ffmpeg = FFMPEG_PATH

import speech_recognition as sr
import wave

#TODO: fix audio shit, create api for the model(easy), queues ?
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}!",
        reply_markup=ForceReply(selective=True),
    )
async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    voice = update.message.voice
    file_id = voice.file_id

    new_file = await context.bot.get_file(file_id)
    cr = await new_file.download_as_bytearray()
    ogg_audio_b = BytesIO(cr)
    with open("audio.ogg", "wb") as f:
        f.write(ogg_audio_b.read())
    # ar = spoken_to_text.AudioRecorder()
    # text = ar.convert_to_text(os.path.join('C:\\Users\\tommy\\OneDrive\\Desktop\\Cearing',"audio.ogg"))
    # print(text)
    ogg_audio = AudioSegment.from_file(file="audio.ogg")
    wav_data = ogg_audio.export(format="wav").read()
    await update.message.reply_text("okey")



async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Echo the user message."""
    txt = update.message.text
    await update.message.reply_text(txt)


def main():
    application = Application.builder().token('5993731273:AAHnf_LvkKJO-oWdXf_O20eDvRXKBF7Bkco').build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("text", echo))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice_message))
    print("Running application...")
    # application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))
    application.run_polling()


if __name__ == "__main__":
    main()