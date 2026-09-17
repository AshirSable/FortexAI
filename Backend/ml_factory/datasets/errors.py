class FileFormatNotValid(Exception):
    def __init__(self, got_format: str, formats_available: list | set | tuple = []):
        self.formats_available = formats_available
        self.got_format = got_format

        super().__init__(self._custom_message)

    @property
    def _custom_message(self) -> str:
        msg = f"File format received: {self.got_format!r}."
        if self.formats_available:
            available_str = ", ".join(map(str, self.formats_available))
            msg += f" Available formats: [{available_str}]."
        return msg
