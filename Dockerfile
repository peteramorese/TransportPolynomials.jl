ARG JULIA_VERSION=1.10.3
FROM julia:${JULIA_VERSION}-bookworm

WORKDIR /app

# Copy dependency files 
COPY Project.toml Manifest.toml ./

# Install all Julia dependencies 
RUN julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Precompile the project
RUN julia --project=. -e 'using Pkg; Pkg.precompile()'

# Copy the rest of the project
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY test/ ./test/

# Default: run Julia with the project environment
ENTRYPOINT ["julia", "--project=."]
CMD ["--help"]
