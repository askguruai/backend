import urllib.parse

import typer
from typing_extensions import Annotated


def build_query_url(
    token: str,
    color: Annotated[
        str,
        typer.Option(
            help="Color to use for popup button, user messages and reaction buttons on hover. Should be in a format of 6 symbols, like 1167B1"
        ),
    ] = "",
    whitelabel: Annotated[
        bool,
        typer.Option(
            help="If whitelable turned on, there will be no AskGuru footer as well as AskGuru icon in a header"
        ),
    ] = True,
    window_heading: Annotated[str, typer.Option(help="Text in the header of a chat")] = "Chat with us!",
    popup_message: Annotated[
        str, typer.Option(help="Text for a popup when chat is not even shown. Use <b>html</b> here!")
    ] = "<b>Chat</b> with us!",
    popup_icon: Annotated[str, typer.Option(help="URL for an icon for a popup and a heading")] = "",
    welcome_message: Annotated[
        str, typer.Option(help="First message to be shown in dialogue. Use **markdown** here.")
    ] = "Hi! I'm AskGuru AI Assistant. Nice to meet you! 👋 Ask me anything... ",
    add_unread_dot: Annotated[bool, typer.Option(help="If to add decorative unread dot on a popup")] = False,
    base_url: str = "https://data.askguru.ai/i",
):
    if not base_url.endswith("?"):
        base_url += "?"

    params = {
        "token": token,
        "whitelabel": whitelabel,
        "windowHeading": window_heading,
        "popupMessage": popup_message,
        "welcomeMessage": welcome_message,
        "addUnreadDot": add_unread_dot,
    }

    if color:
        params |= {"color": color}

    if popup_icon:
        params |= {"popupIcon": popup_icon}

    query_string = urllib.parse.urlencode(params)
    full_url = base_url + query_string

    script_to_insert = f'<script src="{full_url}"></script>'

    typer.echo(script_to_insert)


if __name__ == "__main__":
    typer.run(build_query_url)
