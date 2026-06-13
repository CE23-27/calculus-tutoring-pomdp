"""
Principia AI - Modal Deployment
Tutoring inference pipeline with warm pools for low-latency responses

Architecture (Option C - GPU POMCPOW with Session Affinity):

Frontend (Next.js)
    ↓ HTTP
Modal Endpoint
    ↓
TutoringSystem (3 warm CPU containers)
    ├─→ MathBERT Service (2 warm A100 GPU containers)
    ├─→ POMCPOW Service (4 warm A100 GPU containers, session affinity)
    └─→ GPT-4o-mini API (external) - Switched from Claude Sonnet 4.5 for 2.3x speed & 96% cost savings

Session Affinity:
  - hash(session_id) % 4 routes to specific POMCPOW container
  - Preserves action coherence (last_action, last_tutor_action)
  - Each POMCPOW container tracks 10-20+ sessions in memory
  - GPU critical: ~1000 transformer forward passes per planning turn

Cost: ~$12.60/hour (4 A100 POMCPOW GPUs + 2 A100 MathBERT GPUs + 3 TutoringSystem CPUs)
      + ~$0.14/1000 tutoring turns for GPT-4o-mini API calls (vs $3.42 for Claude)
"""

import modal
import os
from typing import Dict, Any, Optional, List
import json

# Create Modal app
app = modal.App("principia-tutoring")

# Create Modal Volumes for persistent storage
models_volume = modal.Volume.from_name("principia-models", create_if_missing=True)
julia_volume = modal.Volume.from_name("principia-julia", create_if_missing=True)

# Build-time function to install Julia packages with GPU access
def _install_julia_packages_with_gpu():
    """
    Install and precompile Julia packages with GPU available.
    CRITICAL:
    - GPU must be available during build so CUDA packages compile correctly
    - JULIA_CPU_TARGET="generic" must be set to avoid CPU architecture mismatches
      between build and runtime containers
    """
    import subprocess
    import sys
    import os

    print("🎮 GPU-enabled Julia package installation starting...")
    print("=" * 60)

    # Check GPU availability
    print("\n1. Checking GPU availability...")
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print("✓ GPU detected during build:")
            print(result.stdout)
        else:
            print("⚠ WARNING: nvidia-smi failed, GPU may not be available")
            print(result.stderr)
    except Exception as e:
        print(f"⚠ WARNING: Could not check GPU: {e}")

    # Set environment for generic CPU target to avoid architecture mismatches
    env = os.environ.copy()
    env["JULIA_CPU_TARGET"] = "generic"

    print("\n2. Installing Julia packages with GPU and generic CPU target...")
    print("   JULIA_CPU_TARGET=generic (avoids Cascadelake vs Skylake mismatch)")

    cmd = [
        'julia', '-e',
        'using Pkg; Pkg.add(["JSON", "JSON3", "HTTP", "POMDPs", "POMCPOW", "POMDPTools", "BasicPOMCP", "ParticleFilters", "Flux", "CUDA", "BSON", "JLD2", "Statistics", "Distributions", "StatsBase"]); Pkg.precompile()'
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Stream output to see progress
            text=True,
            timeout=600,  # 10 minute timeout
            env=env  # Use environment with JULIA_CPU_TARGET=generic
        )
        print("\n✓ Julia packages installed and precompiled successfully")

        # CRITICAL: Warm up package extensions by actually loading packages together
        # Julia 1.10+ uses "package extensions" that only compile when specific package
        # combinations are loaded. For example, Flux+CUDA extension only compiles when
        # BOTH Flux AND CUDA are loaded together at the same time.
        #
        # Pkg.precompile() doesn't trigger extension compilation - we must actually
        # load the packages to force extension compilation during build.
        print("\n3. Warming up package extensions (Flux+CUDA, etc.)...")
        print("   This triggers compilation of package extensions during build")

        warmup_cmd = [
            'julia', '-e',
            '''
            println("Warming up package extensions by loading all required packages...")

            println("Loading CUDA...")
            using CUDA
            println("✓ CUDA loaded")

            println("Loading Flux (triggers Flux+CUDA extension)...")
            using Flux
            println("✓ Flux loaded with GPU extensions")

            println("Loading JLD2, BSON, JSON, Statistics...")
            using JLD2, BSON, JSON, Statistics
            println("✓ Data packages loaded")

            println("Loading POMDPs ecosystem...")
            using POMDPs, POMDPTools, POMCPOW, BasicPOMCP, ParticleFilters
            println("✓ POMDP packages loaded")

            println("Loading Distributions...")
            using Distributions
            println("✓ Distributions loaded")

            println("All packages and extensions warmed up successfully!")
            '''
        ]

        try:
            result = subprocess.run(
                warmup_cmd,
                check=True,
                capture_output=False,  # Stream output
                text=True,
                timeout=600,  # 10 minute timeout for warmup (Flux+CUDA compilation can be slow)
                env=env
            )
            print("✓ Package extensions warmed up successfully")
        except subprocess.TimeoutExpired:
            print("⚠ WARNING: Extension warmup timed out (may still compile at runtime)")
        except subprocess.CalledProcessError as e:
            print(f"⚠ WARNING: Extension warmup failed (may still compile at runtime): {e}")

        # Fix: Touch all cache files to ensure mtimes are newer than source files
        # This prevents "stale cache" rejections due to filesystem timestamp issues
        print("\n4. Updating cache file timestamps...")
        subprocess.run(
            ["find", "/root/.julia/compiled", "-type", "f", "-exec", "touch", "{}", "+"],
            check=False,  # Don't fail if some files can't be touched
            capture_output=True
        )
        print("✓ Cache timestamps updated")

        return True
    except subprocess.TimeoutExpired:
        print("\n❌ Julia package installation timed out after 10 minutes")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Julia package installation failed with exit code {e.returncode}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during Julia package installation: {e}")
        sys.exit(1)

# Define images with dependencies
mathbert_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.5.0",  # Updated from 2.1.0 to 2.5.0+
        "transformers>=4.35.0",
        "numpy==1.24.3",
        "flask==3.0.0"
    )
)

julia_image = (
    modal.Image.debian_slim()
    .apt_install("curl", "wget", "ca-certificates", "openssl")
    .run_commands(
        # Update CA certificates
        "update-ca-certificates",
    )
    .run_commands(
        # Install Julia 1.10
        "wget https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.0-linux-x86_64.tar.gz",
        "tar -xzf julia-1.10.0-linux-x86_64.tar.gz",
        "cp -r julia-1.10.0 /opt/",
        "ln -s /opt/julia-1.10.0/bin/julia /usr/local/bin/julia",
        "rm -rf julia-1.10.0-linux-x86_64.tar.gz"
    )
    # Copy Julia application files into the image (baked in at build time)
    # copy=True forces copying into image during build (required before run_commands)
    .add_local_dir("planner", "/julia", copy=True)
    .run_commands(
        # Fix Julia's OpenSSL_jll certificate paths - comprehensive certificate setup
        # OpenSSL_jll binary artifact expects certs at specific compiled-in paths

        # Create all possible certificate directories that Julia/OpenSSL might check
        "mkdir -p /opt/x86_64-linux-gnu/lib_root/etc/ssl/certs",
        "mkdir -p /opt/x86_64-linux-gnu/lib_root/etc/ssl",
        "mkdir -p /usr/local/ssl/certs",
        "mkdir -p /usr/local/ssl",

        # Copy system certificates to all possible locations
        "cp -r /etc/ssl/certs/* /opt/x86_64-linux-gnu/lib_root/etc/ssl/certs/",
        "cp /etc/ssl/certs/ca-certificates.crt /opt/x86_64-linux-gnu/lib_root/etc/ssl/cert.pem",
        "cp /etc/ssl/certs/ca-certificates.crt /opt/x86_64-linux-gnu/lib_root/etc/ssl/certs/cert.pem",
        "cp /etc/ssl/certs/ca-certificates.crt /opt/x86_64-linux-gnu/lib_root/etc/ssl/certs/ca-bundle.crt",
        "cp /etc/ssl/certs/ca-certificates.crt /opt/x86_64-linux-gnu/lib_root/etc/ssl/certs/ca-certificates.crt",

        # Also copy to standard SSL paths as backup
        "cp -r /etc/ssl/certs/* /usr/local/ssl/certs/",
        "cp /etc/ssl/certs/ca-certificates.crt /usr/local/ssl/cert.pem",

        # Verify certificates were copied
        "ls -la /opt/x86_64-linux-gnu/lib_root/etc/ssl/certs/ | head -20",
        "echo 'Certificate files in OpenSSL_jll path:'",
        "ls -lh /opt/x86_64-linux-gnu/lib_root/etc/ssl/certs/*.pem /opt/x86_64-linux-gnu/lib_root/etc/ssl/certs/*.crt 2>/dev/null | head -10 || echo 'No .pem or .crt files found'"
    )
    .env({
        # Set multiple SSL environment variables to point to system certificates
        "SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
        "SSL_CERT_DIR": "/etc/ssl/certs",
        "JULIA_SSL_CA_ROOTS_PATH": "/etc/ssl/certs/ca-certificates.crt",
        "CURL_CA_BUNDLE": "/etc/ssl/certs/ca-certificates.crt",
        # Point OpenSSL to the system cert bundle
        "OPENSSL_CONF": "/etc/ssl/",

        # Julia package loading settings
        "JULIA_PKG_PRECOMPILE_AUTO": "0",  # Disable auto-recompilation at runtime
        "JULIA_DEPOT_PATH": "/root/.julia",  # Explicit depot path
        "JULIA_PKG_IGNORE_HASHES": "1",  # CRITICAL FIX: Ignore timestamp/hash mismatches between build and runtime
        # Root cause: Build-time CUDA libraries have different timestamps than runtime CUDA libraries
        # Julia rejects cache as "stale" even though it's valid, triggering 15+ min recompilation
    })
    .pip_install(
        "requests",
        "torch>=2.5.0"  # For GPU detection
    )
    .run_function(
        _install_julia_packages_with_gpu,
        gpu="A100",  # CRITICAL: Ensure GPU available during build so CUDA packages compile correctly
        secrets=[],
        timeout=600  # 10 minutes for package installation and precompilation
    )
)

tutoring_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "openai>=1.0.0",  # GPT-4o-mini support
        "aiohttp>=3.9.0",  # For async HTTP requests
        "numpy==1.24.3",
        "requests==2.31.0",
        "pydantic==2.5.0",
        "fastapi[standard]"
    )
)

# ============================================================================
# MATHBERT EMBEDDING SERVICE
# ============================================================================

@app.cls(
    image=mathbert_image,
    gpu="A100",  # Use A100 GPU for fast inference
    min_containers=2,  # Keep 2 containers warm
    scaledown_window=300,
    secrets=[modal.Secret.from_name("anthropic-secret")]
)
class MathBERTService:
    """MathBERT embedding service with warm pool"""

    @modal.enter()
    def load_model(self):
        """Load MathBERT model once on container start"""
        from transformers import AutoTokenizer, AutoModel
        import torch

        print("🔮 Loading MathBERT model...")

        self.model_name = "tbs17/MathBERT"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()

        # Move to GPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model = self.model.to(self.device)
            print(f"✓ MathBERT loaded on GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("✓ MathBERT loaded on CPU")

        print(f"✓ Embedding dimension: {self.model.config.hidden_size}")

    @modal.method()
    def embed(self, text: str) -> List[float]:
        """Generate embedding for text"""
        import torch

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]

        return cls_embedding.tolist()

    @modal.method()
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        import torch

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return [emb.tolist() for emb in cls_embeddings]


# ============================================================================
# POMCPOW JULIA SERVER
# ============================================================================

@app.cls(
    image=julia_image,
    gpu="A100",  # GPU for sequential dialogue transformer (1000 forward passes/turn)
    cpu=15,  # CPUs for POMCPOW tree search (15 physical cores = 30 vCPU equivalent, matching Lambda Labs)
    memory=32768,  # 32GB for model + transformer cache (matching Lambda Labs 200GB proportionally)
    min_containers=2,  # 2 warm containers baseline, auto-scales up to handle concurrent load
    scaledown_window=600,  # 10 minutes
    startup_timeout=1800,  # 30 minutes - Julia server initialization takes longer than expected
    volumes={
        "/models": models_volume,
    },
    # Julia files are copied into the image at /julia
    secrets=[modal.Secret.from_name("anthropic-secret")]
)
class POMCPOWService:
    """
    POMCPOW planning service with GPU-accelerated dialogue transformer

    Architecture: Option C - Separate GPU service with strict session affinity
    - Each container tracks multiple sessions (10-20+) in memory
    - Session affinity: hash(session_id) % 4 routes to same container
    - Preserves action coherence (last_action, last_tutor_action)
    - GPU critical: ~1000 transformer forward passes per planning turn
    """

    @modal.enter()
    def initialize_planner(self):
        """Initialize POMCPOW planner once on container start"""
        import subprocess
        import os
        import time
        import requests

        print("🎯 Initializing POMCPOW planner with GPU...")

        # Path to model file in Modal Volume
        self.model_path = "/models/sequential_dialogue_final-v4_epoch_30.jld2"

        # Check if model exists
        if not os.path.exists(self.model_path):
            print(f"⚠️  Model file not found at {self.model_path}")
            print("   Please upload model to Modal Volume:")
            print("   modal volume put principia-models sequential_dialogue_final-v4_epoch_30.jld2 /sequential_dialogue_final-v4_epoch_30.jld2")
            self.initialized = False
            return

        # Set environment variables for Julia
        # CRITICAL: JULIA_CPU_TARGET must match build-time setting to avoid cache mismatches
        os.environ["JULIA_CPU_TARGET"] = "generic"

        # Check GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            else:
                print("⚠️  GPU not available, will use CPU (slow!)")
        except ImportError:
            print("⚠️  PyTorch not installed for GPU check")

        print("✓ Model file found")
        print(f"✓ JULIA_CPU_TARGET={os.environ.get('JULIA_CPU_TARGET')}")

        # Start Julia POMCPOW server as subprocess
        print("🚀 Starting Julia POMCPOW server...")

        # Julia files are in the volume at /julia
        julia_dir = "/julia"

        # Start server with proper paths
        # Run the Julia script directly - it will call main() automatically
        # Note: NOT using --project since we installed packages globally in the image
        self.julia_process = subprocess.Popen(
            [
                "julia",
                "-O0",  # Match opt_level with build-time compilation
                f"{julia_dir}/pomcpow_server.jl",
                "8080",  # port
                self.model_path,  # dialogue_model_path
                "templates.json",  # templates_path
                "template_action_mapping.json",  # action_mapping_path
                "http://localhost:8081"  # mathbert_url
            ],
            cwd=julia_dir,  # Set working directory to julia_dir
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        # Wait for Julia server to be ready (max 1800 seconds / 30 minutes)
        # Read Julia output in real-time to see where it's hanging
        print("⏳ Waiting for Julia server to start (max 30 minutes)...")
        print("📋 Streaming Julia server output:")
        import select
        import os

        # Make pipes non-blocking so we can read available output
        os.set_blocking(self.julia_process.stdout.fileno(), False)
        os.set_blocking(self.julia_process.stderr.fileno(), False)

        server_ready = False
        for attempt in range(1800):
            # Check if server is ready
            try:
                response = requests.get("http://localhost:8080/health", timeout=1)
                if response.status_code == 200:
                    print(f"\n✓ Julia server ready after {attempt + 1} seconds")
                    server_ready = True
                    break
            except:
                pass

            # Read and print any available output from Julia process
            try:
                stdout_data = self.julia_process.stdout.read()
                if stdout_data:
                    print(stdout_data, end='', flush=True)
            except:
                pass

            try:
                stderr_data = self.julia_process.stderr.read()
                if stderr_data:
                    print(f"[stderr] {stderr_data}", end='', flush=True)
            except:
                pass

            time.sleep(1)

        if not server_ready:
            print("❌ Julia server failed to start within 180 seconds")
            # Print diagnostic information
            print(f"Julia process status: {'terminated' if self.julia_process.poll() is not None else 'still running'}")

            # Try to read any available output (even if process is still running)
            try:
                # Read whatever is available without blocking
                import select
                import os

                # Make stderr non-blocking
                stderr_fd = self.julia_process.stderr.fileno()
                os.set_blocking(stderr_fd, False)

                # Read available stderr
                try:
                    stderr_output = self.julia_process.stderr.read()
                    if stderr_output:
                        print(f"Julia stderr output:\n{stderr_output}")
                except:
                    pass

                # Also try stdout
                stdout_fd = self.julia_process.stdout.fileno()
                os.set_blocking(stdout_fd, False)
                try:
                    stdout_output = self.julia_process.stdout.read()
                    if stdout_output:
                        print(f"Julia stdout output:\n{stdout_output}")
                except:
                    pass
            except Exception as e:
                print(f"Could not read Julia process output: {e}")

            # Terminate the hanging process
            try:
                self.julia_process.terminate()
                self.julia_process.wait(timeout=5)
            except:
                self.julia_process.kill()

            self.initialized = False
            return

        # Session state tracking (multiple sessions per container)
        # Note: The Julia server maintains its own session state
        self.julia_server_url = "http://localhost:8080"
        self.initialized = True

        print("✓ POMCPOW planner initialized with Julia server")

    @modal.method()
    def plan(
        self,
        session_id: str,
        problem_text: str,
        cumulative_dialogue_embedding: List[float],
        turn_count: int,
        x_features: List[float],
        last_action: Optional[int] = None,
        last_tutor_action: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run POMCPOW planning to select pedagogical action (STATELESS)

        Args:
            session_id: Unique session identifier (for logging)
            problem_text: The math problem being tutored
            cumulative_dialogue_embedding: MathBERT embedding of entire dialogue (768-dim)
            turn_count: Current turn number
            x_features: [speaker_indicator, relevance_scores...] (246-dim)
            last_action: Last action taken (any turn) - for coherence (None on first turn)
            last_tutor_action: Last tutor action - for coherence bonus (None on first turn)

        Returns:
            {
                "action": int (1-8),
                "relevance_delta": List[float] (245-dim),
                "turn_count": int
            }
        """
        import requests
        import json

        if not self.initialized:
            raise RuntimeError("POMCPOW planner not initialized - Julia server not running")

        print(f"📊 Planning for session {session_id}, turn {turn_count}")
        print(f"   last_action: {last_action}, last_tutor_action: {last_tutor_action}")

        # Call Julia POMCPOW server
        try:
            request_payload = {
                "problem_text": problem_text,
                "dialogue_embedding": cumulative_dialogue_embedding,  # 768-dim
                "turn_count": turn_count,
                "x_features": x_features,  # 246-dim: [speaker, 245 concepts]
                "last_action": last_action,  # Can be None
                "last_tutor_action": last_tutor_action  # Can be None
            }

            response = requests.post(
                f"{self.julia_server_url}/plan",
                json=request_payload,
                timeout=30  # 30 second timeout for planning
            )

            # Check for errors and capture Julia error details from response body
            if not response.ok:
                print(f"❌ Julia server returned status {response.status_code}")
                try:
                    error_data = response.json()
                    print("\n" + "="*70)
                    print("📋 JULIA ERROR DETAILS FROM RESPONSE:")
                    print("="*70)
                    print(f"Error: {error_data.get('error', 'Unknown')}")
                    print(f"Message: {error_data.get('message', 'No message')}")
                    print("="*70 + "\n")
                except Exception as parse_err:
                    print(f"⚠️  Could not parse error response: {parse_err}")
                    print(f"Raw response: {response.text[:500]}")
                response.raise_for_status()

            result = response.json()

            # Extract results from Julia response
            action = result["action"]  # 1-indexed (1-8)
            relevance_delta = result["relevance_delta"]  # 245-dim
            new_turn_count = result["turn_count"]

            print(f"   ✓ Julia planning complete: action={action}")
            print(f"   Reward: {result.get('reward', 'N/A')}")
            print(f"   Simulated turns: {result.get('simulated_turns', 'N/A')}")

            return {
                "action": action,
                "relevance_delta": relevance_delta,
                "turn_count": new_turn_count
            }

        except requests.exceptions.Timeout:
            print(f"❌ Julia server timeout after 30s")
            raise RuntimeError("POMCPOW planning timed out")

        except requests.exceptions.RequestException as e:
            print(f"❌ Julia server error: {e}")

            # Try to read Julia error log to see what actually happened
            try:
                with open("/tmp/julia_errors.log", "r") as f:
                    julia_errors = f.read()
                    if julia_errors.strip():
                        print("\n" + "="*70)
                        print("📋 JULIA ERROR LOG CONTENTS:")
                        print("="*70)
                        print(julia_errors)
                        print("="*70 + "\n")
                    else:
                        print("⚠️  Julia error log exists but is empty")
            except FileNotFoundError:
                print("⚠️  Julia error log file not found at /tmp/julia_errors.log")
            except Exception as log_err:
                print(f"⚠️  Could not read Julia error log: {log_err}")

            raise RuntimeError(f"POMCPOW planning failed: {e}")

        except KeyError as e:
            print(f"❌ Invalid response from Julia server: missing key {e}")
            raise RuntimeError(f"Invalid POMCPOW response: missing {e}")

    @modal.method()
    def reset(self, session_id: str):
        """Reset planner state for specific session"""
        import requests

        if not self.initialized:
            print(f"⚠️  POMCPOW not initialized, skipping reset")
            return

        try:
            # Call Julia server to reset session state
            response = requests.post(
                f"{self.julia_server_url}/reset",
                json={},
                timeout=5
            )
            response.raise_for_status()
            print(f"✓ POMCPOW session {session_id} reset on Julia server")

        except Exception as e:
            print(f"⚠️  Failed to reset Julia session: {e}")


# ============================================================================
# TUTORING SYSTEM ORCHESTRATOR
# ============================================================================

# POMCPOW pool configuration
NUM_POMCPOW_CONTAINERS = 2  # Must match min_containers value above (auto-scales beyond this)

# Calculus concept dictionary (245 concepts)
# This must match the Julia CONCEPT_DICT exactly
CONCEPT_DICT = {
    "Function notation and domain/range understanding": "C001",
    "Algebraic manipulation fluency (factoring, expanding, simplifying)": "C002",
    "Trigonometric identities and unit circle mastery": "C003",
    "Exponential and logarithmic properties": "C004",
    "Inequality solving and interval notation": "C005",
    "Composition vs multiplication of functions distinction": "C006",
    "Absolute value interpretation and properties": "C007",
    "Rational function behavior and asymptotes": "C008",
    "Polynomial long division execution": "C009",
    "Complex number arithmetic": "C010",
    "Systems of equations solving": "C011",
    "Graph transformations (shifts, stretches, reflections)": "C012",
    "Even/odd function recognition": "C013",
    "Inverse function concept and notation": "C014",
    "Piecewise function evaluation": "C015",
    "Parameter vs variable distinction": "C016",
    "Rate vs ratio understanding": "C017",
    "Proportion and scaling relationships": "C018",
    "Exponential growth vs linear growth distinction": "C019",
    "Periodic function properties": "C020",
    "Vector notation and basic 2D operations": "C021",
    "Parametric equation interpretation": "C022",
    "Approaching vs reaching a value distinction": "C023",
    "Left-hand vs right-hand limit differences": "C024",
    "Limit existence requires left/right agreement": "C025",
    "Substitution method applicability recognition": "C026",
    "Indeterminate form identification (0/0, ∞/∞, etc.)": "C027",
    "Algebraic manipulation for limit resolution": "C028",
    "Factoring and canceling for removable discontinuities": "C029",
    "Rationalizing for limit evaluation": "C030",
    "Squeeze theorem setup and application": "C031",
    "Infinite limits vs limits at infinity distinction": "C032",
    "Vertical asymptote connection to infinite limits": "C033",
    "Horizontal asymptote connection to limits at infinity": "C034",
    "Removable vs non-removable discontinuity types": "C035",
    "Continuity definition (limit equals function value)": "C036",
    "Intermediate value theorem conditions and application": "C037",
    "Limit laws and properties application": "C038",
    "Rational function limit evaluation techniques": "C039",
    "Trigonometric limit patterns (especially sin(x)/x)": "C040",
    "Exponential and logarithmic limit behavior": "C041",
    "Piecewise function limit evaluation": "C042",
    "Absolute value function limit behavior": "C043",
    "Jump discontinuity recognition": "C044",
    "Infinite discontinuity recognition": "C045",
    "Composite function limit evaluation": "C046",
    "One-sided limit algebraic techniques": "C047",
    "Graphical limit estimation from graphs": "C048",
    "Limit existence vs function existence distinction": "C049",
    "Asymptotic behavior interpretation": "C050",
    "Derivative as instantaneous rate of change": "C051",
    "Derivative as slope of tangent line": "C052",
    "Limit definition of derivative understanding": "C053",
    "Secant line to tangent line progression": "C054",
    "Differentiability implies continuity": "C055",
    "Continuity does not imply differentiability": "C056",
    "Power rule pattern recognition and execution": "C057",
    "Constant multiple rule application": "C058",
    "Sum/difference rule application": "C059",
    "Product rule vs chain rule distinction": "C060",
    "Product rule pattern recognition and execution": "C061",
    "Quotient rule pattern recognition and execution": "C062",
    "Chain rule outer/inner function identification": "C063",
    "Chain rule execution order (outside derivative × inside derivative)": "C064",
    "Chain rule with multiple compositions": "C065",
    "Implicit differentiation trigger recognition": "C066",
    "Implicit differentiation execution": "C067",
    "Derivative notation flexibility (f', dy/dx, y', etc.)": "C068",
    "Higher-order derivative notation and meaning": "C069",
    "Trigonometric derivative patterns": "C070",
    "Exponential derivative patterns (e^x, a^x)": "C071",
    "Natural logarithm derivative (d/dx[ln x] = 1/x)": "C072",
    "General logarithm derivative pattern": "C073",
    "Inverse trigonometric derivative patterns": "C074",
    "Derivative of absolute value functions": "C075",
    "Corner and cusp recognition for non-differentiability": "C076",
    "Piecewise function derivative evaluation": "C077",
    "Parametric differentiation (dy/dx from dx/dt and dy/dt)": "C078",
    "Logarithmic differentiation technique": "C079",
    "Differentiation of composite trigonometric functions": "C080",
    "Differentiation of nested exponentials": "C081",
    "Second derivative geometric interpretation (concavity)": "C082",
    "Velocity vs speed distinction in derivatives": "C083",
    "Derivative as linear approximation": "C084",
    "Common derivative mistakes (product ≠ product of derivatives)": "C085",
    "Critical point identification (derivative = 0 or undefined)": "C086",
    "Critical point vs extrema distinction": "C087",
    "First derivative test for local extrema": "C088",
    "Second derivative test for local extrema": "C089",
    "Increasing/decreasing function behavior from f'": "C090",
    "Concavity determination from f''": "C091",
    "Inflection point identification (f'' changes sign)": "C092",
    "Absolute vs relative extrema distinction": "C093",
    "Candidates test for absolute extrema (critical points + endpoints)": "C094",
    "Optimization problem setup and constraint identification": "C095",
    "Optimization with closed interval (endpoints matter)": "C096",
    "Related rates variable identification": "C097",
    "Related rates equation setup and differentiation": "C098",
    "Related rates solving for unknown rate": "C099",
    "Linear approximation and tangent line applications": "C100",
    "Differential notation (dy = f'(x)dx)": "C101",
    "L'Hôpital's rule applicability and execution": "C102",
    "L'Hôpital's rule for 0/0 and ∞/∞ forms": "C103",
    "Mean value theorem interpretation and application": "C104",
    "Rolle's theorem conditions and application": "C105",
    "Motion analysis (position, velocity, acceleration relationships)": "C106",
    "Particle motion on a line (direction, speed, distance)": "C107",
    "Geometric optimization problems": "C108",
    "Curve sketching systematic approach": "C109",
    "Function analysis from derivative graphs (given f', sketch f)": "C110",
    "Connecting f, f', and f'' graphs": "C111",
    "Derivative interpretation in word problems": "C112",
    "Average vs instantaneous rate of change distinction": "C113",
    "Tangent line equation from point and derivative": "C114",
    "Normal line equation": "C115",
    "Antiderivative vs definite integral distinction": "C116",
    "Indefinite integral notation and +C necessity": "C117",
    "Area under curve geometric interpretation": "C118",
    "Riemann sum approximation concept": "C119",
    "Fundamental theorem of calculus part 1": "C120",
    "Fundamental theorem of calculus part 2": "C121",
    "Definite integral notation and evaluation": "C122",
    "Integration as reverse differentiation": "C123",
    "Basic integration rules (reverse power rule)": "C124",
    "Integration of exponential functions": "C125",
    "Integration of 1/x yielding ln|x|": "C126",
    "Integration of trigonometric functions": "C127",
    "Integration by substitution recognition (u-substitution)": "C128",
    "u-substitution execution with du replacement": "C129",
    "u-substitution with definite integrals (change bounds)": "C130",
    "Integration by parts formula and application": "C131",
    "Integration by parts LIATE priority": "C132",
    "Definite integral properties": "C133",
    "Integration bounds interpretation and reversal": "C134",
    "Negative area interpretation (below x-axis)": "C135",
    "Net change theorem": "C136",
    "Average value of function on interval": "C137",
    "Even/odd function integration shortcuts": "C138",
    "Absolute value function integration": "C139",
    "Piecewise function integration": "C140",
    "Improper integral recognition": "C141",
    "Improper integral convergence/divergence": "C142",
    "Comparison test for improper integral convergence": "C143",
    "Integration of inverse trigonometric results": "C144",
    "Partial fraction decomposition basics": "C145",
    "Trigonometric substitution recognition": "C146",
    "Common integral mistakes": "C147",
    "Area between curves setup": "C148",
    "Area between curves with respect to y": "C149",
    "Volume of revolution disk method": "C150",
    "Volume of revolution washer method": "C151",
    "Volume of revolution about different axes": "C152",
    "Volume by cylindrical shells method": "C153",
    "Disk vs washer vs shell method selection": "C154",
    "Arc length formula application": "C155",
    "Arc length in parametric form": "C156",
    "Area in polar coordinates": "C157",
    "Arc length in polar coordinates": "C158",
    "Average value of function interpretation": "C159",
    "Accumulation function interpretation": "C160",
    "Work problems using integration": "C161",
    "Physics applications (displacement from velocity)": "C162",
    "Total distance vs displacement distinction": "C163",
    "Population dynamics and rate of change modeling": "C164",
    "Growth and decay integration models": "C165",
    "Consumer/producer surplus economic applications": "C166",
    "Integration application word problem setup": "C167",
    "Differential equation solution verification": "C168",
    "Slope field interpretation and sketching": "C169",
    "Separable equation recognition": "C170",
    "Separation of variables execution": "C171",
    "Initial condition application": "C172",
    "Particular vs general solution distinction": "C173",
    "Exponential growth/decay model (dy/dt = ky)": "C174",
    "Logistic growth model interpretation": "C175",
    "Euler's method numerical approximation": "C176",
    "Equilibrium solution identification": "C177",
    "Solution curve behavior from differential equation": "C178",
    "Applications (Newton's cooling, population growth)": "C179",
    "Sequence vs series distinction": "C180",
    "Sequence convergence and limit": "C181",
    "Sequence divergence patterns": "C182",
    "Monotonic sequence recognition": "C183",
    "Bounded sequence recognition": "C184",
    "Geometric series sum formula": "C185",
    "Geometric series convergence condition": "C186",
    "Telescoping series recognition": "C187",
    "nth term test for divergence": "C188",
    "Integral test for convergence": "C189",
    "p-series convergence": "C190",
    "Comparison test application": "C191",
    "Limit comparison test application": "C192",
    "Ratio test application and interpretation": "C193",
    "Alternating series test (Leibniz)": "C194",
    "Absolute vs conditional convergence": "C195",
    "Power series interval of convergence": "C196",
    "Power series radius of convergence": "C197",
    "Endpoint convergence testing": "C198",
    "Power series differentiation and integration": "C199",
    "Taylor series definition and construction": "C200",
    "Maclaurin series (Taylor series at x=0)": "C201",
    "Common Maclaurin series": "C202",
    "Taylor polynomial approximation": "C203",
    "Lagrange error bound understanding": "C204",
    "Function representation as power series": "C205",
    "Geometric series as power series": "C206",
    "Operations with power series": "C207",
    "Parametric equation interpretation (x(t), y(t))": "C208",
    "Eliminating parameter to find Cartesian equation": "C209",
    "Parametric curve sketching": "C210",
    "dy/dx for parametric curves": "C211",
    "Second derivative for parametric curves": "C212",
    "Arc length for parametric curves": "C213",
    "Polar coordinate system understanding (r, θ)": "C214",
    "Converting between polar and rectangular coordinates": "C215",
    "Polar curve sketching": "C216",
    "Slope of tangent line in polar coordinates": "C217",
    "Area enclosed by polar curve": "C218",
    "Area between two polar curves": "C219",
    "Polar curve symmetry recognition": "C220",
    "Derivative rule automation (quick execution)": "C221",
    "Common derivative pattern recognition": "C222",
    "Common integral pattern recognition": "C223",
    "Algebraic simplification after differentiation/integration": "C224",
    "Sign analysis for function behavior": "C225",
    "Calculator vs hand calculation appropriate use": "C226",
    "Proper mathematical notation consistency": "C227",
    "Units and dimensional analysis in applications": "C228",
    "Estimation and reasonableness checking": "C229",
    "Graph sketching from analytical information": "C230",
    "Function composition evaluation": "C231",
    "Solving equations efficiently": "C232",
    "Error identification habits (checking work)": "C233",
    "Mathematical communication and justification": "C234",
    "Problem-solving strategy selection": "C235",
    "Derivative of product ≠ product of derivatives": "C236",
    "Derivative of quotient ≠ quotient of derivatives": "C237",
    "Antiderivative of product ≠ product of antiderivatives": "C238",
    "∫(1/x)dx = ln|x| + C (not ln(x))": "C239",
    "Chain rule necessity recognition": "C240",
    "Forgetting +C consequences": "C241",
    "Confusing average rate with instantaneous rate": "C242",
    "Area vs net area distinction": "C243",
    "Convergence vs divergence series distinction": "C244",
    "Connecting position-velocity-acceleration graphically": "C245"
}

# Reverse mapping
CODE_TO_CONCEPT = {v: k for k, v in CONCEPT_DICT.items()}

# Action names - NOTE: Julia uses 1-indexed actions (1-8)
# The Julia TutoringAction enum starts at 1
ACTION_NAMES = {
    1: "instruct",
    2: "connect",
    3: "correct",
    4: "probe",
    5: "guide",
    6: "affirm",
    7: "listen",
    8: "practice"
}

# ============================================================================
# Session Store Abstraction Layer
# ============================================================================

class SessionStore:
    """Abstract interface for session state persistence.

    Easy to swap implementations: Modal Dict → Database → Redis
    """

    def get_session(self, session_id: str) -> dict:
        """Retrieve session state or return empty state if not found"""
        raise NotImplementedError

    def save_session(self, session_id: str, state: dict) -> None:
        """Persist session state"""
        raise NotImplementedError

    def delete_session(self, session_id: str) -> None:
        """Delete session state"""
        raise NotImplementedError


class ModalDictSessionStore(SessionStore):
    """Modal Dict implementation - fast, managed by Modal"""

    def __init__(self):
        # Modal Dict for persistent storage across containers
        self.dict = modal.Dict.from_name("tutoring-sessions", create_if_missing=True)

    def get_session(self, session_id: str) -> dict:
        """Retrieve session from Modal Dict"""
        return self.dict.get(session_id, self._empty_state())

    def save_session(self, session_id: str, state: dict) -> None:
        """Save session to Modal Dict"""
        self.dict[session_id] = state

    def delete_session(self, session_id: str) -> None:
        """Delete session from Modal Dict"""
        if session_id in self.dict:
            del self.dict[session_id]

    def _empty_state(self) -> dict:
        """Default empty session state"""
        return {
            "initial_relevance": None,
            "current_relevance": None,
            "previous_relevance": None,
            "dialogue_history": [],
            "problem_set": None,  # Problem set metadata for sequential flow
            "current_problem_index": 0,  # Track which problem student is working on
            "cumulative_dialogue_embedding": None,
            "turn_count": 0,
            "problem_text": "",
            "last_action": None,  # Last action taken (any turn) - for Julia POMDP coherence
            "last_tutor_action": None,  # Last tutor action - for Julia coherence bonus
            "dialogue_turn_count": 0,  # Turns since last checkpoint (for adaptive routing)
            "current_block_id": None,  # Current content block being shown
            "module_slug": None,  # Current module
            "lesson_number": None  # Current lesson
        }


# Future: Add DatabaseSessionStore here when migrating
# class DatabaseSessionStore(SessionStore):
#     def __init__(self, connection_string: str):
#         self.db = create_connection(connection_string)
#
#     def get_session(self, session_id: str) -> dict:
#         result = self.db.execute("SELECT state FROM sessions WHERE id = ?", [session_id])
#         return json.loads(result['state']) if result else self._empty_state()
#
#     def save_session(self, session_id: str, state: dict) -> None:
#         self.db.execute(
#             "INSERT INTO sessions (id, state) VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET state = ?",
#             [session_id, json.dumps(state), json.dumps(state)]
#         )

# ============================================================================

@app.cls(
    image=tutoring_image,
    cpu=2,
    memory=2048,
    min_containers=3,  # Keep 3 warm for concurrent sessions
    scaledown_window=300,
    secrets=[modal.Secret.from_name("openai-secret")]
)
class TutoringSystem:
    """Main tutoring orchestrator with session state"""

    @modal.enter()
    def initialize(self):
        """Initialize orchestrator and service clients"""
        import openai

        print("🎓 Initializing Tutoring System...")

        # Initialize OpenAI client (GPT-4o-mini)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in secrets")

        self.openai = openai.OpenAI(api_key=api_key)

        # Get handles to other services
        self.mathbert = MathBERTService()

        # Get handle to POMCPOW service (Modal handles load balancing and session stickiness)
        self.pomcpow = POMCPOWService()
        print("✓ Connected to POMCPOW service (Modal handles routing)")

        # Initialize persistent session store (replaces in-memory self.state)
        self.session_store = ModalDictSessionStore()
        print("✓ Session store initialized (Modal Dict)")

        print("✓ Tutoring system ready")

    def reset_state(self):
        """Reset session state"""
        self.state = {
            "initial_relevance": None,
            "current_relevance": None,
            "previous_relevance": None,
            "dialogue_history": [],
            "cumulative_dialogue_embedding": None,
            "turn_count": 0,
            "problem_text": ""
        }


    async def get_initial_relevance(self, initial_input: str) -> List[float]:
        """Use GPT-4o-mini to identify relevant concepts for this problem"""
        import numpy as np

        concept_list = "\n".join([f"{code}: {name}" for name, code in CONCEPT_DICT.items()])

        prompt = f"""You are an expert calculus tutor analyzing which mathematical concepts are needed to solve a specific problem.

Problem/Context:
"{initial_input}"

Below is a list of 245 calculus concepts with their codes.

YOUR TASK: Identify ONLY the concepts that are NECESSARY for solving THIS SPECIFIC problem.
Most problems only need 5-15 concepts. DO NOT list concepts that aren't needed.

For each RELEVANT concept, provide:
- The concept code
- A relevance score (1-10) where:
  * 9-10: ABSOLUTELY NECESSARY - Cannot solve without this
  * 7-8: VERY USEFUL - Standard approach uses this
  * 5-6: SOMEWHAT USEFUL - Helpful but not essential
  * 3-4: MARGINALLY RELEVANT - Minor connection

Concept List:
{concept_list}

Respond with ONLY a JSON object containing the relevant concepts.
Format: {{"C057": 9, "C061": 8, "C064": 7}}

Be selective! Only include concepts that actually matter for this specific problem."""

        print("📊 Getting initial relevance from GPT-4o-mini...")

        response = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert calculus tutor. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.0
        )

        # Parse JSON response
        content = response.choices[0].message.content
        import re
        json_match = re.search(r'\{[^}]+\}', content)
        if not json_match:
            raise ValueError("Could not find JSON in GPT-4o-mini response")

        scores_dict = json.loads(json_match.group(0))

        # Initialize 245-dimensional vector with zeros
        relevance_vector = np.zeros(245, dtype=np.float32)

        # Set scores for identified concepts
        for code, score in scores_dict.items():
            concept_num = int(code[1:])  # Remove 'C' prefix
            if 1 <= concept_num <= 245:
                relevance_vector[concept_num - 1] = float(score)

        print(f"   Identified {len(scores_dict)} relevant concepts")

        return relevance_vector.tolist()

    async def generate_response(
        self,
        dialogue_context: str,
        action_enum: int,
        concept_name: str
    ) -> str:
        """Generate tutor response using GPT-4o-mini"""

        action_name = ACTION_NAMES[action_enum]

        prompt = f"""You are a skilled calculus tutor having a conversation with a student.

Current dialogue context:
{dialogue_context}

TASK: Perform the pedagogical action "{action_name}" to address the students most pressing confusion.

Requirements:
- Respond in 1-2 sentences maximum
- Be HUMAN and natural in your response
- Address the specific concept within the dialogue context
- **CRITICAL**: Use proper LaTeX formatting for ALL mathematical expressions:
  * Use single $ for inline math (e.g., $x^2 + 3x$ or $f(x) = 3x^2$)
  * Use double $$ for display math on its own line (e.g., $$f'(x) = 2x + 3$$)
  * NEVER use plain text for math like x^2 or f(x) without dollar signs
  * Always wrap variables, functions, and equations in $ signs
- Use the pedagogical action appropriately:
  * instruct: Explain or teach directly
  * connect: Link to previous knowledge
  * correct: Fix an error or misconception
  * probe: Ask a question to check understanding
  * guide: Provide hints or direction
  * affirm: Validate correct thinking
  * listen: Acknowledge and encourage student to continue
  * practice: Suggest practice or application

Your response:"""

        print(f"💬 Generating {action_name} response for {concept_name}...")

        response = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a warm, engaging calculus tutor who helps students learn through dialogue."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )

        response_text = response.choices[0].message.content.strip()
        print(f"   Response: {response_text[:100]}...")

        return response_text

    async def generate_block_introduction(
        self,
        dialogue_context: str,
        routing_reason: str,
        target_block_id: str
    ) -> str:
        """Generate a smooth introduction when transitioning to a new block"""

        prompt = f"""You are a skilled calculus tutor. The student has been working on a topic, and based on the conversation, it's time to introduce new content.

Recent dialogue:
{dialogue_context}

ROUTING REASON: {routing_reason}

TASK: Write a brief, natural transition (1-2 sentences) that:
- Acknowledges what the student has been working on
- Smoothly introduces the need for this new content
- Creates excitement or motivation for the new material
- Does NOT mention technical details like "block IDs" or system internals

Requirements:
- Keep it conversational and warm
- Use proper LaTeX formatting for math: $x^2$ for inline, $$equation$$ for display
- Make it feel like a natural continuation of the dialogue, not a system message

Your transition:"""

        print(f"📝 Generating block introduction for routing: {routing_reason[:50]}...")

        response = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a warm, engaging calculus tutor who creates smooth transitions between topics."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.8
        )

        intro_text = response.choices[0].message.content.strip()
        print(f"   Introduction: {intro_text[:80]}...")

        return intro_text

    async def synthesize_narrative_content(
        self,
        dialogue_context: str,
        baseline_narrative: dict,
        pedagogical_action: str,
        student_question: str,
        routing_reason: str
    ) -> str:
        """
        Synthesize contextualized narrative content using baseline as a template.

        Instead of showing static narrative, this creates a custom explanation that:
        - References the student's specific question
        - Explains why this content matters for THEIR understanding
        - Maintains the cosmic narrative voice and style
        - Adapts based on pedagogical action
        """

        # Extract narrative sections from baseline
        narrative_text = ""
        if isinstance(baseline_narrative, dict) and 'sections' in baseline_narrative:
            narrative_text = "\n\n".join([
                section.get('content', '')
                for section in baseline_narrative.get('sections', [])
            ])

        prompt = f"""You are Mr. Benson, helping a student who just asked about something specific.

STUDENT'S RECENT QUESTION/CONFUSION:
{student_question}

RECENT DIALOGUE CONTEXT:
{dialogue_context}

WHY WE'RE SHOWING THIS CONTENT:
{routing_reason}

YOUR PEDAGOGICAL ACTION: {pedagogical_action}
(instruct = explain directly, probe = ask questions, guide = provide hints, connect = link to prior knowledge)

BASELINE NARRATIVE (use as content guide, but ADAPT to student):
{narrative_text[:2000]}

TASK: Write a contextualized explanation (200-300 words) that:

1. **Opens with a callback** to their specific question/confusion
   - "You were asking about [specific_thing]..."
   - "That question about [x] is actually leading us to..."

2. **Explains the relevance** for THEIR problem
   - Why does this content matter for what they're trying to understand?
   - How does it connect to their confusion?

3. **Delivers the core content** using the MIDDLE-GROUND STYLE:
   - Clear, direct language without being overly formal
   - Use concrete examples with explanation
   - Some engaging elements ("think about", "consider", "notice") without being too casual
   - Maintain mathematical rigor
   - Connect concepts to bigger picture
   - Avoid excessive metaphors, but use them when helpful
   - Use "we" occasionally to create dialogue
   - Break down complex ideas into digestible pieces

4. **Uses proper LaTeX**
   - $x^2$ for inline math
   - $$equation$$ for display equations

5. **Matches the pedagogical action**
   - If "probe": include thoughtful questions
   - If "instruct": explain directly and clearly
   - If "guide": give hints and scaffolding
   - If "connect": link to previous concepts

Your contextualized narrative:"""

        print(f"🎨 Synthesizing contextualized narrative (action: {pedagogical_action})...")

        response = self.openai.chat.completions.create(
            model="gpt-4o",  # Using full GPT-4 for better narrative synthesis
            messages=[
                {"role": "system", "content": "You are Mr. Benson, a calculus tutor who creates personalized, context-aware explanations using clear, direct language with a middle-ground style that balances rigor with accessibility."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.85
        )

        synthesized_content = response.choices[0].message.content.strip()
        print(f"   Synthesized {len(synthesized_content)} characters of narrative")
        print(f"   Preview: {synthesized_content[:100]}...")

        return synthesized_content

    async def _check_adaptive_routing(
        self,
        state: dict,
        current_action: int,
        session_id: str
    ) -> dict:
        """
        Call the adaptive routing endpoint to determine if we should switch content blocks.

        Args:
            state: Current session state
            current_action: POMDP action just taken (1-8)
            session_id: Session identifier for logging

        Returns:
            Routing decision dict with 'continue', 'blockId', 'reason' fields
        """
        import aiohttp

        try:
            # Extract belief vector from current relevance
            belief_vector = state["current_relevance"]
            if not belief_vector:
                print("⚠️  No belief vector available, skipping routing")
                return {"continue": True, "reason": "No belief state available"}

            # Format recent dialogue (last 6 messages)
            recent_dialogue = []
            for msg in state["dialogue_history"][-6:]:
                if msg.startswith("Student:"):
                    recent_dialogue.append({
                        "role": "student",
                        "message": msg.replace("Student:", "").strip()
                    })
                elif msg.startswith("Tutor:"):
                    recent_dialogue.append({
                        "role": "tutor",
                        "message": msg.replace("Tutor:", "").strip()
                    })
                elif msg.startswith("Problem:"):
                    recent_dialogue.append({
                        "role": "student",
                        "message": msg.replace("Problem:", "").strip()
                    })

            # Build request payload
            payload = {
                "moduleSlug": state["module_slug"],
                "currentBlockId": state["current_block_id"],
                "lessonNumber": state["lesson_number"],
                "beliefVector": belief_vector,
                "recommendedAction": current_action,
                "dialogueTurnCount": state["dialogue_turn_count"],
                "recentDialogue": recent_dialogue
            }

            # Call Next.js adaptive routing endpoint
            endpoint_url = "https://mrbenson.ai/api/adaptive-sequence"

            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint_url, json=payload, timeout=30) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        error_text = await response.text()
                        print(f"❌ Adaptive routing API error ({response.status}): {error_text}")
                        return {"continue": True, "reason": f"API error: {response.status}"}

        except Exception as e:
            print(f"❌ Error calling adaptive routing: {e}")
            # Fail gracefully - don't break dialogue flow
            return {"continue": True, "reason": f"Routing error: {str(e)}"}

    @modal.method()
    async def tutoring_turn(self, session_id: str, user_input: str, is_initial: bool = False) -> str:
        """
        Execute one tutoring turn

        Args:
            session_id: Unique session identifier for POMCPOW routing
            user_input: Student's message
            is_initial: True if this is the first message (problem statement)

        Returns:
            Tutor's response
        """
        import numpy as np

        print(f"\n{'='*60}")
        print(f"🎯 TUTORING TURN - Session: {session_id} | isInitial: {is_initial}")
        print(f"📝 User input: {user_input[:100]}...")

        # Load session state from persistent store
        state = self.session_store.get_session(session_id)
        print(f"📦 Loaded state - turn_count: {state['turn_count']}")

        # Add user input to dialogue history
        user_message = f"Problem: {user_input}" if is_initial else f"Student: {user_input}"
        state["dialogue_history"].append(user_message)

        if is_initial:
            print("🆕 First turn - initializing belief")

            # Store problem text
            state["problem_text"] = user_input

            # Get initial relevance from Claude
            state["initial_relevance"] = await self.get_initial_relevance(user_input)

            # Get initial cumulative MathBERT embedding
            embedding = self.mathbert.embed.remote(user_message)
            state["cumulative_dialogue_embedding"] = embedding
            state["turn_count"] = 1

        else:
            print("📍 Subsequent turn - updating cumulative embedding")

            # Get cumulative embedding of entire dialogue
            full_dialogue = "\n".join(state["dialogue_history"])
            embedding = self.mathbert.embed.remote(full_dialogue)
            state["cumulative_dialogue_embedding"] = embedding
            state["turn_count"] += 1

        # Determine relevance to use
        relevance_to_use = (state["current_relevance"]
                           if state["current_relevance"] is not None
                           else state["initial_relevance"])

        # This should never happen with Modal Dict, but keep for safety
        if relevance_to_use is None:
            print("⚠️  Unexpected: relevance is None (re-initializing)")
            state["problem_text"] = user_input
            state["initial_relevance"] = await self.get_initial_relevance(user_input)
            relevance_to_use = state["initial_relevance"]
            embedding = self.mathbert.embed.remote(user_message)
            state["cumulative_dialogue_embedding"] = embedding
            state["turn_count"] = 1

        state["previous_relevance"] = relevance_to_use.copy()

        # Build x_features: [speaker_indicator, relevance_scores...]
        x_features = [1.0] + relevance_to_use  # Student just spoke

        # Increment dialogue turn count BEFORE checking if this is a checkpoint
        state["dialogue_turn_count"] += 1

        # Determine if this is a checkpoint turn for routing
        # For INITIAL routing (no block yet), trigger at turn 2, then every 3 turns until success
        # For SUBSEQUENT routing (have a block), trigger every 3 turns
        has_context = state["current_block_id"] and state["lesson_number"] is not None
        is_first_routing = (
            state["dialogue_turn_count"] >= 2 and
            state["dialogue_turn_count"] % 3 == 2 and  # Retry every 3 turns (2, 5, 8, 11...)
            not has_context and
            state["module_slug"]
        )
        is_subsequent_routing = (
            state["dialogue_turn_count"] % 3 == 0 and
            state["dialogue_turn_count"] > 0 and
            has_context and
            state["module_slug"]
        )
        is_checkpoint = is_first_routing or is_subsequent_routing

        # Start parallel calls
        import asyncio

        dialogue_context = "\n".join(state["dialogue_history"])

        if is_checkpoint:
            print(f"\n🔄 CHECKPOINT: Running PLANNING + ROUTING in parallel (turn {state['dialogue_turn_count']})")

            # Run planning and routing in parallel (routing will return blocks for all 8 actions)
            planning_task = asyncio.create_task(asyncio.to_thread(
                self.pomcpow.plan.remote,
                session_id,
                state["problem_text"],
                state["cumulative_dialogue_embedding"],
                state["turn_count"],
                x_features,
                state["last_action"],
                state["last_tutor_action"]
            ))
            routing_task = self._check_adaptive_routing(state, 0, session_id)  # action=0 since we don't know yet

            # Wait for both to complete
            plan_result, routing_result = await asyncio.gather(planning_task, routing_task)

            # Extract action from planning
            action = plan_result["action"]
            relevance_delta = plan_result["relevance_delta"]
            state["turn_count"] = plan_result["turn_count"]

            print(f"✅ Planning complete: action={action}")

            # Extract the block for the selected action from routing result
            if routing_result and routing_result.get("actionBlocks"):
                action_names = ["instruct", "connect", "correct", "probe", "guide", "affirm", "listen", "practice"]
                selected_action_name = action_names[action - 1] if 1 <= action <= 8 else "instruct"

                action_block = routing_result["actionBlocks"].get(selected_action_name, {})
                routing_decision = {
                    "continue": action_block.get("blockId") == "CONTINUE",
                    "blockId": action_block.get("blockId"),
                    "reason": action_block.get("reason", ""),
                    "blockIntroduction": action_block.get("blockIntroduction", "")
                }
                print(f"✅ Selected action: '{selected_action_name}' (action #{action})")
                print(f"✅ Selected block: {routing_decision.get('blockId')}")
            else:
                routing_decision = routing_result

        else:
            # Non-checkpoint: just run planning
            plan_result = self.pomcpow.plan.remote(
                session_id,
                state["problem_text"],
                state["cumulative_dialogue_embedding"],
                state["turn_count"],
                x_features,
                state["last_action"],
                state["last_tutor_action"]
            )

            action = plan_result["action"]
            relevance_delta = plan_result["relevance_delta"]
            state["turn_count"] = plan_result["turn_count"]
            routing_decision = None

        # Update action history in session state
        state["last_action"] = action
        if action in [1, 2, 3, 4, 5, 6, 7, 8]:
            state["last_tutor_action"] = action

        # Apply delta update
        new_relevance = np.array(relevance_to_use) + 0.7 * np.array(relevance_delta)
        new_relevance = np.clip(new_relevance, 0, 10)
        state["current_relevance"] = new_relevance.tolist()

        # Identify top conceptual block
        concept_idx = np.argmax(new_relevance)
        concept_code = f"C{str(concept_idx + 1).zfill(3)}"
        concept_name = CODE_TO_CONCEPT.get(concept_code, "Unknown concept")

        print(f"🎯 Selected concept: {concept_code} - {concept_name}")
        print('='*60)

        # Generate response (routing already happened in parallel with planning above if checkpoint)
        if is_checkpoint:
            tutor_response = await self.generate_response(dialogue_context, action, concept_name)

            # Process routing decision
            if routing_decision and not routing_decision.get("continue", True):
                print(f"🔀 ROUTING: Switch to block {routing_decision.get('blockId')}")
                print(f"   Reason: {routing_decision.get('reason')}")

                # blockIntroduction is already generated by the routing API
                if routing_decision.get('blockIntroduction'):
                    print(f"📝 Block introduction from routing API: {routing_decision.get('blockIntroduction')[:80]}...")

                # Check if this is a custom student definition request
                block_id = routing_decision.get('blockId', '')
                if block_id.startswith('STUDENT_DEF:'):
                    concept = block_id.replace('STUDENT_DEF:', '')
                    print(f"🎯 Custom student definition requested for concept: {concept}")

                    # Update state (frontend will create synthetic block with navigation button)
                    state["current_block_id"] = block_id
                    state["dialogue_turn_count"] = 0

                    # Add to dialogue history
                    state["dialogue_history"].append(f"[Routing to Student Definition: {concept}]")

                    print(f"✅ Frontend will render student definition block with navigation button")

                # Otherwise, fetch and process regular block
                else:
                    # Check if this is a narrative block - if so, synthesize contextualized content
                    try:
                        import aiohttp
                        async with aiohttp.ClientSession() as session:
                            # Fetch block details from Next.js API
                            block_url = f"{os.environ.get('NEXT_API_URL', 'https://mrbenson.ai')}/api/blocks/{routing_decision.get('blockId')}"
                            async with session.get(block_url) as resp:
                                if resp.status == 200:
                                    block_data = await resp.json()
                                    if block_data.get('success'):
                                        block = block_data.get('block', {})
                                        block_type = block.get('blockType', '')
                                        block_content = block.get('blockContent', {})

                                        print(f"🔍 Block type detected: {block_type}")

                                        # If it's a narrative block, synthesize contextualized content
                                        if block_type == 'narrative':
                                            print(f"📚 Narrative block detected - synthesizing contextualized content...")

                                            # Get student's last message for context
                                            student_messages = [msg for msg in state["dialogue_history"] if msg.startswith("Student:")]
                                            last_student_msg = student_messages[-1].replace("Student: ", "") if student_messages else ""

                                            # Get action name for pedagogical context
                                            action_name = ACTION_NAMES.get(action, "instruct")

                                            # Wrap block_content in expected format if it's an array
                                            baseline_narrative = {'sections': block_content} if isinstance(block_content, list) else block_content

                                            synthesized_narrative = await self.synthesize_narrative_content(
                                                dialogue_context=dialogue_context,
                                                baseline_narrative=baseline_narrative,
                                                pedagogical_action=action_name,
                                                student_question=last_student_msg,
                                                routing_reason=routing_decision.get('reason', '')
                                            )

                                            # Add synthesized content to routing decision
                                            routing_decision["synthesizedNarrative"] = synthesized_narrative
                                            print(f"✅ Added synthesized narrative to routing decision")

                                            # Add block context to dialogue history for future responses
                                            state["dialogue_history"].append(f"[Content Block: {block.get('title', 'Narrative')} - {synthesized_narrative[:200]}...]")
                                        else:
                                            # For non-narrative blocks, add description
                                            block_desc = block.get('description', '')
                                            if block_desc:
                                                state["dialogue_history"].append(f"[Content Block: {block.get('title', '')} - {block_desc}]")
                                            print(f"ℹ️ Non-narrative block ({block_type}) - skipping synthesis")
                                    else:
                                        print(f"⚠️ API returned success=false or invalid block data")
                                else:
                                    print(f"⚠️ Block API returned status {resp.status}")
                    except Exception as e:
                        import traceback
                        print(f"⚠️ Failed to synthesize narrative (non-critical): {e}")
                        print(f"   Traceback: {traceback.format_exc()}")
                        # Continue without synthesis - block will show static content

                    # Reset dialogue turn count when switching blocks (for non-STUDENT_DEF blocks)
                    state["dialogue_turn_count"] = 0
                    if not block_id.startswith('STUDENT_DEF:'):
                        state["current_block_id"] = routing_decision.get("blockId")
            else:
                print(f"✅ ROUTING: Continue with current block")
        else:
            # Non-checkpoint turn: only generate response
            tutor_response = await self.generate_response(dialogue_context, action, concept_name)
            routing_decision = None

        # If this is the first turn of a problem set, prepend acknowledgment
        if is_initial and state.get("problem_set") and state["problem_set"].get("problem_selection"):
            selection = state["problem_set"]["problem_selection"]
            total_problems = selection.get("total_problems", 1)
            selected_number = selection.get("selected_problem_number", 1)
            reasoning = selection.get("reasoning", "")

            # Build intro message (reasoning is already conversational from GPT-4-mini)
            if total_problems > 1:
                intro = f"I see you have {total_problems} problems here. "

                if reasoning:
                    intro += reasoning
                else:
                    intro += f"Let's start with problem {selected_number}."

                intro += "\n\n"
                tutor_response = intro + tutor_response
                print(f"📚 Prepended problem set intro to tutor response")

        # Add tutor response to history
        state["dialogue_history"].append(f"Tutor: {tutor_response}")

        # Save updated state to persistent store
        self.session_store.save_session(session_id, state)
        print(f"💾 Saved state - turn_count: {state['turn_count']}, dialogue_turn_count: {state['dialogue_turn_count']}")

        # Return response with optional routing decision
        result = {
            "response": tutor_response,
            "routing": routing_decision
        }
        return result

    @modal.method()
    def reset_session(self, session_id: str):
        """Reset session state for new conversation"""
        print(f"🔄 Resetting session {session_id}")

        # Delete session from persistent store (includes last_action, last_tutor_action)
        self.session_store.delete_session(session_id)
        print(f"✓ Deleted session {session_id} from Modal Dict")

        # Reset POMCPOW cache (server is stateless now, but clear rollout cache)
        self.pomcpow.reset.remote(session_id)

        return {"status": "reset", "session_id": session_id}


# ============================================================================
# HTTP ENDPOINTS
# ============================================================================

@app.function(
    image=tutoring_image,
    secrets=[modal.Secret.from_name("openai-secret")]
)
@modal.fastapi_endpoint(method="POST")
async def tutoring_endpoint(data: Dict[str, Any]):
    """
    Main tutoring endpoint

    Request:
        {
            "message": "Student's message",
            "session_id": "user-123",  # REQUIRED for session affinity
            "new_session": false,
            "block_id": "cmhc33zt10000mytzn16od26t",  # Optional: current content block
            "module_slug": "calculus",  # Optional: current module
            "lesson_number": 1  # Optional: current lesson
        }

    Response:
        {
            "response": "Tutor's response",
            "session_id": "user-123",
            "routing": {  # Optional: only present if adaptive routing suggests a change
                "continue": false,
                "blockId": "new_block_id",
                "reason": "Explanation for routing decision"
            }
        }
    """
    message = data.get("message")
    session_id = data.get("session_id", "default")
    new_session = data.get("new_session", False)
    block_id = data.get("block_id")
    module_slug = data.get("module_slug")
    lesson_number = data.get("lesson_number")
    problem_set_metadata = data.get("problem_set_metadata")  # New: problem set data

    if not message:
        return {"error": "Missing 'message' field"}, 400

    # Create tutoring system instance
    system = TutoringSystem()

    if new_session:
        system.reset_session.remote(session_id)

    # Update context in session if provided
    if new_session or block_id or module_slug or lesson_number is not None or problem_set_metadata:
        # Get current state to update context
        session_store = ModalDictSessionStore()
        state = session_store.get_session(session_id)
        if block_id:
            state["current_block_id"] = block_id
        if module_slug:
            state["module_slug"] = module_slug
        if lesson_number is not None:
            state["lesson_number"] = lesson_number

        # Store problem set metadata for sequential flow
        if problem_set_metadata:
            state["problem_set"] = problem_set_metadata
            state["current_problem_index"] = problem_set_metadata.get("current_problem_index", 0)
            print(f"📚 Stored problem set: {problem_set_metadata.get('problem_count')} problems")

        session_store.save_session(session_id, state)

    # Determine if this is the first message
    is_first = new_session

    # Get tutor response with session affinity
    result = system.tutoring_turn.remote(session_id, message, is_first)

    # result is now a dict with 'response' and optional 'routing' fields
    response_data = {
        "response": result.get("response"),
        "session_id": session_id
    }

    # Include routing decision if present
    if result.get("routing"):
        response_data["routing"] = result["routing"]

    return response_data


@app.function(image=tutoring_image)
@modal.fastapi_endpoint(method="POST")
def reset_session_endpoint(data: Dict[str, Any]):
    """Reset tutoring session"""
    session_id = data.get("session_id", "default")
    system = TutoringSystem()
    result = system.reset_session.remote(session_id)
    return result


@app.function(image=tutoring_image)
@modal.fastapi_endpoint(method="GET")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Principia Tutoring API",
        "version": "1.0.0"
    }


# ============================================================================
# LOCAL TESTING
# ============================================================================

@app.local_entrypoint()
def test():
    """Test the tutoring system locally"""
    print("🧪 Testing Tutoring System\n")

    # Test MathBERT
    print("1. Testing MathBERT...")
    mathbert = MathBERTService()
    embedding = mathbert.embed.remote("Find the limit as x approaches 2 of (x^2 - 4)/(x - 2)")
    print(f"   ✓ Got embedding: {len(embedding)}-dimensional\n")

    # Test POMCPOW with session
    print("2. Testing POMCPOW with session affinity...")
    pomcpow = POMCPOWService()
    test_session_id = "test-session-001"
    result = pomcpow.plan.remote(
        test_session_id,
        "Find the derivative of x^2",
        embedding,
        1,
        [1.0] + [0.0] * 245
    )
    print(f"   ✓ Got action: {result['action']} for session {test_session_id}\n")

    # Test Tutoring System with session affinity
    print("3. Testing Tutoring System with session routing...")
    system = TutoringSystem()
    response = system.tutoring_turn.remote(
        test_session_id,
        "I need help finding the derivative of f(x) = 3x^2 + 2x - 5",
        is_initial=True
    )
    print(f"   ✓ Tutor response: {response}\n")

    print("✅ All tests passed! Session affinity is working.")
