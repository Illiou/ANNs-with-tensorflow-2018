discount_factor = 0.99
def discount_rewards(rewards, discount_factor):
    discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
    for i, reward in enumerate(reversed(rewards)):
        discounted_rewards[-(i+1)] = discounted_rewards[-i] * discount_factor + reward
        
    normalized_rewards = ((discounted_rewards - np.mean(discounted_rewards)) / np.std(discounted_rewards))
    return normalized_rewards