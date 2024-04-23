from ...model.clustering import cluster_metrics


def generate_sequence(N: int) -> list:
    sequence = []

    # From 1 to 10 with step 1
    sequence.extend(range(1, 11))

    # From 10 to 50 with step 2
    sequence.extend(range(12, 51, 2))

    # From 50 to 200 with step 10
    sequence.extend(range(60, 201, 10))

    # From 200 to N with step of 1% of N
    current = 210 if 200 + 0.01 * N > 200 else 200 + 0.01 * N
    step = max(1, 0.01 * N)  # Ensure at least 1 step if N is less than 100
    while current <= N:
        sequence.append(int(current))
        current += step

    sequence = [x for x in sequence if x <= N]
    return sequence


def debug_step(params, model, df_all, y_train, df_true, y_true) -> dict:
    step = {**params}
    # Compute the clustering metrics for the train and test set
    y_pred = model.fit_predict(df_all)
    res = cluster_metrics(y_train, y_pred[:len(y_train)])
    res = {f'cv_train_{k}': v for k, v in res.items()}
    step.update(res)

    y_pred = model.fit_predict(df_true)
    res = cluster_metrics(y_true, y_pred)
    res = {f'test_{k}': v for k, v in res.items()}
    step.update(res)
    return step


def debug_step_test(params, model, df_all, y_train, df_true, y_true, y_test) -> dict:
    step = {**params}
    # Compute the clustering metrics for the train and test set
    y_pred = model.fit_predict(df_all)

    res = cluster_metrics(y_train, y_pred[:len(y_train)])
    res = {f'cv_train_{k}': v for k, v in res.items()}
    step.update(res)

    res = cluster_metrics(y_test, y_pred[len(y_train):len(y_train) + len(y_test)])
    res = {f'cv_test_{k}': v for k, v in res.items()}
    step.update(res)

    y_pred = model.fit_predict(df_true)
    res = cluster_metrics(y_true, y_pred)
    res = {f'test_{k}': v for k, v in res.items()}
    step.update(res)
    return step
