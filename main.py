from telegram import Update, File, User
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove

from telegram.ext import Updater, Dispatcher, CallbackContext
from telegram.ext import CommandHandler, MessageHandler, Filters, ConversationHandler

from misc import load_model, clear_image, get_transferred_image

TOKEN = '...'  # insert token here

MENU, STYLE, TARGET = range(3)
MODEL = load_model()


def get_keyboard_markup(user_data: dict) -> ReplyKeyboardMarkup:
    reply_keyboard = [[f'{"✔️" if user_data["style"] else "❌"} Set style'],
                      [f'{"✔️" if user_data["target"] else "❌"} Set target'],
                      [f'{"✔️" if (user_data["style"] and user_data["target"]) else "❌"} Make transfer'],
                      ['/stop']]
    return ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=False,
                               input_field_placeholder='Select command')


def start(update: Update, context: CallbackContext) -> int:
    context.user_data['style'] = None
    context.user_data['target'] = None

    keyboard_markup = get_keyboard_markup(context.user_data)

    update.message.reply_text('Hello! I am Style Transfer Bot.\n'
                              'To make a transfer:\n'
                              '- Set the style image\n'
                              '- Set the target image\n'
                              '❌ or ✔️ shows the status of the command.\n'
                              'Send /stop to stop the conversation and reset images.',
                              reply_markup=keyboard_markup)

    return MENU


def set_style(update: Update, context: CallbackContext) -> int:
    update.message.reply_text('Ok. Send me the style image for transfer.',
                              reply_markup=ReplyKeyboardRemove())
    return STYLE


def set_style_image(update: Update, context: CallbackContext) -> int:
    if context.user_data['style']:
        clear_image(context.user_data['style'])

    photo_file: File = update.message.photo[-1].get_file()
    path = photo_file.download(f'images/{photo_file.file_id}.jpg')

    context.user_data['style'] = path

    keyboard_markup = get_keyboard_markup(context.user_data)

    update.message.reply_text('Style image was successfully set!',
                              reply_markup=keyboard_markup)
    return MENU


def set_target(update: Update, context: CallbackContext) -> int:
    update.message.reply_text('Ok. Send me the target image for transfer.',
                              reply_markup=ReplyKeyboardRemove())
    return TARGET


def set_target_image(update: Update, context: CallbackContext) -> int:
    if context.user_data['target']:
        clear_image(context.user_data['target'])

    photo_file: File = update.message.photo[-1].get_file()
    path = photo_file.download(f'images/{photo_file.file_id}.jpg')

    context.user_data['target'] = path

    keyboard_markup = get_keyboard_markup(context.user_data)

    update.message.reply_text('Target image was successfully set!',
                              reply_markup=keyboard_markup)
    return MENU


def make_transfer(update: Update, context: CallbackContext) -> int:
    user: User = update.message.from_user

    style = context.user_data['style']
    target = context.user_data['target']

    if style and target:
        image = get_transferred_image(MODEL, style, target)

        clear_image(context.user_data['style'])
        clear_image(context.user_data['target'])

        context.user_data['style'] = None
        context.user_data['target'] = None

        keyboard_markup = get_keyboard_markup(context.user_data)

        user.send_photo(image, 'Success!', reply_markup=keyboard_markup)
    else:
        update.message.reply_text('Please set style and target image first.\n'
                                  'After that you will see the ✔️ status near commands.')
    return MENU


def stop(update: Update, context: CallbackContext) -> int:
    clear_image(context.user_data['style'])
    clear_image(context.user_data['target'])

    context.user_data['style'] = None
    context.user_data['target'] = None

    update.message.reply_text('Images were reset.\n'
                              'Send /start to start the conversation again.\n'
                              'Bye!', reply_markup=ReplyKeyboardRemove())

    return ConversationHandler.END


def main() -> None:
    updater: Updater = Updater(TOKEN)
    dispatcher: Dispatcher = updater.dispatcher

    conversation_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            MENU: [
                MessageHandler(Filters.regex('^(❌ Set style|✔️ Set style)$'), set_style),
                MessageHandler(Filters.regex('^(❌ Set target|✔️ Set target)$'), set_target),
                MessageHandler(Filters.regex('^(❌ Make transfer|✔️ Make transfer)$'), make_transfer)
            ],
            STYLE: [MessageHandler(Filters.photo, set_style_image)],
            TARGET: [MessageHandler(Filters.photo, set_target_image)],
        },
        fallbacks=[CommandHandler('stop', stop)]
    )
    dispatcher.add_handler(conversation_handler)

    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
