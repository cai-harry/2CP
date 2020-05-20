def print_global_performance(client):
    loss, accuracy = client.evaluate_global()
    print(f"Loss {loss}\tAccuracy {accuracy}")


def print_trainer_performances(client):
    scores = client.evaluate_trainers()
    print(f"\t Scores{scores.values()}\t")


def print_token_count(client):
    tokens, total_tokens = client.get_token_count()
    percent = int(100*tokens/total_tokens) if tokens > 0 else 0
    print(f"\t\t{client.name} has {tokens} of {total_tokens} tokens ({percent}%)")
