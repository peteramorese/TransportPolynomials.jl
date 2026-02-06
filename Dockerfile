ARG JULIA_VERSION=1.10.3
FROM julia:${JULIA_VERSION}-bookworm

WORKDIR /root/TransportPolynomials

# Install Python and matplotlib for PyPlot
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-matplotlib \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files 
COPY Project.toml Manifest.toml ./

# Install all Julia dependencies 
RUN julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Precompile the project
RUN julia --project=. -e 'using Pkg; Pkg.precompile()'

COPY . .
