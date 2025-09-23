



def make_prediction(model: nn.Module, observation: np.array, question: float, device: torch.device = torch.device('cpu')) -> float:
    torch.cuda.empty_cache()
    gc.collect()
    model.to(device).eval()
    observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)
    question_tensor = torch.tensor([[question]], dtype=torch.float32).to(device)
    with torch.no_grad():
        possible_answer, _, _ = model(observation_tensor, question_tensor)
    return possible_answer.item()


def plot_prediction(observation, question, answer, possible_answer) -> None:
    fig = plt.figure(figsize=(10, 6))
    plt.plot(observation, label='Observation', color='blue')
    plt.scatter(question, possible_answer, color='red', label='Prediction', zorder=5)
    plt.scatter(question, answer, color='green', label='True Answer', zorder=5)
    plt.title('Pendulum Prediction')
    plt.xlabel('Time Steps')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.show()
    return None


if __name__ == "__main__":

    N_samples = 1
    kapa_range = (4.5, 5.5)
    b_range = (0.4, 0.6)
    observations, questions, answers, params = build_dataset(N_samples, kapa_range, b_range)

    possible_answer = make_prediction(pendulum_net, observations[0], questions[0], device=device)
    plot_prediction(observations[0], questions[0], answers[0], possible_answer)