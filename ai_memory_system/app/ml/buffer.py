buffer = []
BUFFER_SIZE = 5


def add_to_buffer(embedding, memory_id):
    if any(item[1] == memory_id for item in buffer):
        return

    buffer.append((embedding, memory_id))


def ready_to_train():
    return len(buffer) >= BUFFER_SIZE


def get_buffer():
    return buffer


def clear_buffer():
    global buffer
    buffer = []
