class EmptyReplayBufferException(Exception):
    def __init__(self) -> None:
        super().__init__(f'Experience Replay Buffer is empty.')
