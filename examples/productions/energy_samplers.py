import numpy as np

def dbn_fermi_dirac(x, mu, kt):
    """Fermi-Dirac helper function."""
    arg = (x - mu) / kt
    return 1 / (1 + np.exp(np.clip(arg, -20, 20)))

def get_energies(dist_type, min_e, max_e, n_events):
    """
    Returns an array of energy values based on the requested distribution.
    """
    if dist_type == "linear":
        return np.random.uniform(min_e, max_e, n_events)

    elif dist_type == "log":
        log_min = np.log10(min_e)
        log_max = np.log10(max_e)
        return 10**np.random.uniform(log_min, log_max, n_events)

    elif dist_type == "fermi":
        x = np.linspace(min_e, max_e, 10000)
        
        mu1, kt1 = 0.5, 0.1
        mu2, kt2 = 200, 20
        y0 = 0.005
        
        y1 = dbn_fermi_dirac(x, mu1, kt1)
        y2 = dbn_fermi_dirac(x, mu2, kt2)
        y = y0 - y1 + y2
        
        probs = y / np.sum(y)
        return np.random.choice(x, size=n_events, p=probs)
    
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")