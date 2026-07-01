import numpy as np

# Distributions that can be selected from the command line (i.e. that don't
# require a python callable). Keep this in sync with get_energies() below.
SUPPORTED_DISTRIBUTIONS = ["linear", "log", "fermi"]


def sample_linear(min_e, max_e, n_events):
    """Samples energies from a uniform linear distribution."""
    return np.random.uniform(min_e, max_e, n_events)

def sample_log(min_e, max_e, n_events):
    """Samples energies from a logarithmic distribution."""
    if min_e <= 0:
        raise ValueError("Minimum energy for log distribution must be > 0.")
    log_min = np.log10(min_e)
    log_max = np.log10(max_e)
    return 10**np.random.uniform(log_min, log_max, n_events)

def dbn_fermi_dirac(x, mu, kt):
    """Fermi-Dirac helper function."""
    arg = (x - mu) / kt
    return 1 / (1 + np.exp(np.clip(arg, -20, 20)))

def sample_fermi(min_e, max_e, n_events, mu1=0.5, kt1=0.1, mu2=200, kt2=20, y0=0.005, resolution=10000):
    """Samples energies using the custom Fermi-Dirac equation."""
    x = np.linspace(min_e, max_e, resolution)
    
    y1 = dbn_fermi_dirac(x, mu1, kt1)
    y2 = dbn_fermi_dirac(x, mu2, kt2)
    y = y0 - y1 + y2
    
    y = np.clip(y, 0, None)
    probs = y / np.sum(y)
    
    return np.random.choice(x, size=n_events, p=probs)

def get_energies(dist_type, min_e, max_e, n_events, custom_pdf=None, resolution=10000, **kwargs):
    """
    Returns an array of energy values based on the requested distribution.
    """
    if dist_type == "linear":
        return sample_linear(min_e, max_e, n_events)

    elif dist_type == "log":
        return sample_log(min_e, max_e, n_events)

    elif dist_type == "fermi":
        return sample_fermi(min_e, max_e, n_events, **kwargs)
    
    
    elif dist_type == "custom":
        if custom_pdf is None:
            raise ValueError("You must provide a custom_pdf function")
        x = np.linspace(min_e, max_e, resolution)
        y = custom_pdf(x)
        
        y = np.clip(y, 0, None)
        probs = y / np.sum(y)
        return np.random.choice(x, size=n_events, p=probs)
    
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")