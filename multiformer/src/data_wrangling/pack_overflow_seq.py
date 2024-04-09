def pack_seq(
    example,
    min_seq_len=65,
    max_seq_len=1024,
    delimiter_id=29889,
    eos_id=2,
    para_separator=13,
    sos_id=1,
):
    count = example["len"]
    idx = example["idx"]
    bucket = []
    overflow_bucket = []
    start = 0
    end = len(count) - 1

    while start < end:
        max_seq_len = max_seq_len
        if count[end] >= max_seq_len:
            index = [
                indx
                for indx, i in enumerate(idx[end][:max_seq_len])
                if i == delimiter_id
            ]
            if len(index):
                index = index[-1] + 1
                trim_token = idx[end][:index] + [eos_id]
                bucket.append(trim_token)
                # overflow
                overflow_token = [sos_id] + idx[end][index:]
                if len(overflow_token) > min_seq_len:
                    index = True  # break the while loop there is not delimiter in the text (i.e [])
                    while (len(overflow_token) > max_seq_len) and index:
                        index = [
                            indx
                            for indx, i in enumerate(overflow_token[:max_seq_len])
                            if i == delimiter_id
                        ]
                        if len(index):
                            index = index[-1] + 1
                            trim_token = overflow_token[:index] + [eos_id]
                            overflow_bucket.append(trim_token)
                            overflow_token = [sos_id] + overflow_token[index:]
                    overflow_bucket.append(overflow_token)
            else:
                print("Discarding sample at index : ", end)

        else:
            if count[end] + count[start] > max_seq_len:
                bucket.append(idx[end])
            else:
                small_bucket = []
                idx[end].pop()  # remove eod_id
                idx[start].pop()
                small_bucket.extend(idx[end])
                small_bucket.extend(
                    [para_separator, para_separator]
                )  # adding /n/n after each entry
                small_bucket.extend(idx[start])

                # max_seq_len -= 2
                while (start < end) and (
                    len(small_bucket) + count[start + 1] <= (max_seq_len - 2)
                ):
                    start += 1
                    # max_seq_len -= 2
                    small_bucket.extend([para_separator, para_separator])
                    idx[start].pop()
                    small_bucket.extend(idx[start])
                small_bucket.append(eos_id)
                bucket.append(small_bucket)
                start += 1
        end -= 1
    if start == end:
        bucket.append(idx[end])

    bucket = bucket + [None] * (
        len(idx) - len(bucket)
    )  # hf dataset lib expect map func to return same dim as input i.e len(bucket) == len(idx)
    overflow_bucket = overflow_bucket + [None] * (len(idx) - len(overflow_bucket))
    return {"packed": bucket, "overflow": overflow_bucket}
