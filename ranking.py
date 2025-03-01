import curses
def interactive_ranking(choices):
    def main(stdscr):
        curses.curs_set(0)  # Hide cursor
        selected = 0  # Index of the selected item
        down = False

        while True:
            stdscr.clear()
            stdscr.addstr("Use ↑ ↓ to move, Enter to confirm ranking\n\n", curses.A_BOLD)

            for i, choice in enumerate(choices):
                if i == selected:
                    stdscr.addstr(f"> {choice}\n", curses.A_REVERSE)
                else:
                    stdscr.addstr(f"  {choice}\n")

            key = stdscr.getch()
            if key == curses.KEY_UP and selected > 0:
                if down:
                    choices[selected], choices[selected - 1] = choices[selected - 1], choices[selected]
                selected -= 1
            elif key == curses.KEY_DOWN and selected < len(choices) - 1:
                if down:
                    choices[selected], choices[selected + 1] = choices[selected + 1], choices[selected]
                selected += 1
            elif key == 32:  # Enter key
                down = not down

            elif key == 10:  # Escape key
                break
    curses.wrapper(main)
    return choices
