def batcher(iterable, batch_size=50):
    """
    Yield lists of items from 'iterable' in batches of size 'batch_size'.
    """
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:  # yield remaining items
        yield batch
