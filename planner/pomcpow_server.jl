# #!/usr/bin/env julia
# """
# HTTP Server for POMCPOW Algebra Tutoring Planner
# Receives requests from Python, runs planning, returns action and relevance
# """

# using HTTP
# using JSON
# using POMDPs
# using POMCPOW
# using Statistics

# # Include the POMDP definition
# include("POMDP.jl")  # Your main POMDP file

# # Global planner state
# mutable struct PlannerState
#     pomdp::Union{Nothing, AlgebraTutoringPOMDP}
#     planner::Union{Nothing, Any}
#     updater::Union{Nothing, TutoringBeliefUpdater}
#     current_belief::Union{Nothing, TutoringState}
#     dialogue_transformer::Union{Nothing, Any}
#     device::Union{Nothing, Function}
#     mathbert_embedding::Union{Nothing, Vector{Float32}}
# end

# const PLANNER_STATE = PlannerState(nothing, nothing, nothing, nothing, nothing, nothing, nothing)

# function initialize_planner(model_path::String="dialogue_transformer_trained.bson")
#     """Initialize the POMCPOW planner (done once at startup)"""
#     println("Initializing POMCPOW planner...")
    
#     # Load dialogue transformer
#     PLANNER_STATE.dialogue_transformer, PLANNER_STATE.device = load_dialogue_model(model_path)
    
#     # Initialize templates
#     initialize_all_templates!()
    
#     # Create POMDP with placeholder problem text
#     PLANNER_STATE.pomdp = AlgebraTutoringPOMDP(
#         dialogue_transformer = PLANNER_STATE.dialogue_transformer,
#         device = PLANNER_STATE.device,
#         problem_text = ""  # Will be set per request
#     )
    
#     # Create solver
#     solver = create_pomcpow_solver(PLANNER_STATE.pomdp, n_iterations=100)
#     PLANNER_STATE.planner = solve(solver, PLANNER_STATE.pomdp)
    
#     # Create updater
#     PLANNER_STATE.updater = TutoringBeliefUpdater(PLANNER_STATE.pomdp)
    
#     println("✓ Planner initialized and ready")
# end

# function handle_plan_request(req::HTTP.Request)
#     """Handle /plan POST request"""
#     try
#         # Parse request body
#         body = JSON.parse(String(req.body))
#         dialogue_context = body["dialogue_context"]
#         x_features = Float32.(body["x_features"])
        
#         # 🆕 Get MathBERT embedding if provided (Turn 1 only)
#         mathbert_embedding = haskey(body, "mathbert_embedding") ? 
#                             Float32.(body["mathbert_embedding"]) : nothing
        
#         println("\n" * "="^60)
#         println("Received planning request")
#         println("Dialogue length: $(length(dialogue_context)) chars")
        
#         # Extract components from x_features (271-dimensional)
#         # x_features format: [speaker_indicator, relevance_scores...]
#         # Index 1: speaker_indicator (1.0 for student, 0.0 for tutor)
#         # Index 2-271: relevance scores [0-10]
#         speaker_indicator = x_features[1]
#         relevance = x_features[2:271]
        
#         println("Speaker indicator: $(speaker_indicator)")
#         println("Relevance mean: $(round(mean(relevance), digits=2))")
        
#         # 🆕 Handle MathBERT embedding on first turn
#         if mathbert_embedding !== nothing
#             # Cache it for the entire session
#             PLANNER_STATE.mathbert_embedding = mathbert_embedding
#         end
        
#         # Create or update belief state
#         if PLANNER_STATE.current_belief === nothing
#             # First turn - initialize
#             PLANNER_STATE.current_belief = TutoringState(
#                 relevance,
#                 dialogue_context,
#                 Int[],
#                 nothing
#             )
#             println("Initialized first belief state")
#         else
#             # Subsequent turn - update with new context and relevance
#             PLANNER_STATE.current_belief = TutoringState(
#                 relevance,
#                 dialogue_context,
#                 PLANNER_STATE.current_belief.template_history,
#                 PLANNER_STATE.current_belief.last_action
#             )
#             println("Updated belief state (turn $(length(PLANNER_STATE.current_belief.template_history) + 1))")
#         end
        
#         # Plan action using POMCPOW
#         println("Running POMCPOW planning...")
#         action = POMDPs.action(PLANNER_STATE.planner, PLANNER_STATE.current_belief)
#         action_enum = Int(action)
        
#         println("Selected action: $(ACTION_NAMES[action_enum]) (enum: $action_enum)")
        
#         # Execute action to get next state and observation
#         sp, o, r = execute_action_in_environment(
#             PLANNER_STATE.pomdp,
#             PLANNER_STATE.current_belief,
#             action
#         )
        
#         # Update belief for next turn
#         PLANNER_STATE.current_belief = POMDPs.update(
#             PLANNER_STATE.updater,
#             PLANNER_STATE.current_belief,
#             action,
#             o
#         )
        
#         # Get current relevance from observation and apply ReLU + asymptotic scaling
#         # This ensures relevance is always in [0, 10] range when returned to client
#         current_relevance_raw = o.relevance_observations
        
#         # Apply bounded scaling: ReLU then asymptotic to [0, 10]
#         current_relevance_positive = max.(0.0f0, current_relevance_raw)  # ReLU
#         current_relevance = 10.0f0 .* (current_relevance_positive ./ (current_relevance_positive .+ 3.0f0))  # Asymptotic scaling to [0,10]
        
#         println("Updated relevance mean: $(round(mean(current_relevance), digits=2))")
#         println("Reward: $(round(r, digits=2))")
        
#         # Prepare response
#         response_data = Dict(
#             "action" => action_enum,
#             "current_relevance" => current_relevance,
#             "reward" => r,
#             "turn" => length(PLANNER_STATE.current_belief.template_history)
#         )
        
#         return HTTP.Response(200, JSON.json(response_data))
        
#     catch e
#         println("ERROR: $e")
#         showerror(stdout, e, catch_backtrace())
#         error_response = Dict("error" => string(e))
#         return HTTP.Response(500, JSON.json(error_response))
#     end
# end

# function handle_reset_request(req::HTTP.Request)
#     """Handle /reset POST request to start new session"""
#     try
#         println("\n" * "="^60)
#         println("RESETTING PLANNER STATE FOR NEW SESSION")
#         println("="^60)
        
#         # Clear current belief
#         PLANNER_STATE.current_belief = nothing
        
#         # 🆕 Clear cached MathBERT embedding
#         PLANNER_STATE.mathbert_embedding = nothing
        
#         # Clear transformer cache
#         if PLANNER_STATE.pomdp !== nothing
#             cache_size = length(PLANNER_STATE.pomdp.transformer_cache)
#             empty!(PLANNER_STATE.pomdp.transformer_cache)
#             println("✓ Cleared transformer cache ($cache_size entries)")
#         end
        
#         println("✓ Cleared MathBERT embedding cache")  # 🆕
#         println("✓ Reset complete")
#         println("="^60 * "\n")
        
#         return HTTP.Response(200, JSON.json(Dict("status" => "reset")))
#     catch e
#         println("ERROR during reset: $e")
#         showerror(stdout, e, catch_backtrace())
#         error_response = Dict("error" => string(e))
#         return HTTP.Response(500, JSON.json(error_response))
#     end
# end

# function handle_health_request(req::HTTP.Request)
#     """Handle /health GET request"""
#     status = Dict(
#         "status" => "healthy",
#         "planner_initialized" => PLANNER_STATE.planner !== nothing,
#         "current_turn" => PLANNER_STATE.current_belief !== nothing ? 
#                          length(PLANNER_STATE.current_belief.template_history) : 0
#     )
#     return HTTP.Response(200, JSON.json(status))
# end

# # Router
# function router(req::HTTP.Request)
#     if req.method == "POST" && req.target == "/plan"
#         return handle_plan_request(req)
#     elseif req.method == "POST" && req.target == "/reset"
#         return handle_reset_request(req)
#     elseif req.method == "GET" && req.target == "/health"
#         return handle_health_request(req)
#     else
#         return HTTP.Response(404, "Not Found")
#     end
# end

# function main()
#     """Start the HTTP server"""
#     println("\n" * "="^60)
#     println("POMCPOW Algebra Tutoring Server")
#     println("="^60)
    
#     # Parse command line arguments
#     port = 8080
#     model_path = "dialogue_transformer_trained.bson"
    
#     if length(ARGS) >= 1
#         port = parse(Int, ARGS[1])
#     end
#     if length(ARGS) >= 2
#         model_path = ARGS[2]
#     end
    
#     # Initialize planner
#     initialize_planner(model_path)
    
#     # Start server
#     println("\nStarting HTTP server on port $port...")
#     println("Endpoints:")
#     println("  POST /plan   - Plan next action")
#     println("  POST /reset  - Reset for new session")
#     println("  GET  /health - Health check")
#     println("\nServer ready. Press Ctrl+C to stop.")
#     println("="^60 * "\n")
    
#     HTTP.serve(router, "0.0.0.0", port)
# end

# # Action name mapping for logging
# const ACTION_NAMES = Dict(
#     1 => "instruct",
#     2 => "connect",
#     3 => "correct",
#     4 => "probe",
#     5 => "guide",
#     6 => "affirm",
#     7 => "listen",
#     8 => "practice"
# )

# # Run server if executed directly
# if abspath(PROGRAM_FILE) == @__FILE__
#     main()
# end

"""
HTTP Server for POMCPOW Calculus Tutoring Planner
Sequential Dialogue Transformer with Delta-Based Belief Updates
Uses MathBERT for first turn, lightweight encoder for planning simulations
"""

# Using simple Sockets-based HTTP server instead of HTTP.jl to avoid SSL issues
# Print immediately to confirm script execution
println("🚀 POMCPOW SERVER: Script starting...")
flush(stdout)

include("simple_http_server.jl")

println("✓ Loaded simple_http_server.jl")
flush(stdout)

using JSON
println("✓ Loaded JSON")
flush(stdout)

using POMDPs
using POMCPOW

using Statistics
println("✓ Loaded Statistics")
flush(stdout)

using JLD2
println("✓ Loaded JLD2")
flush(stdout)

using Flux
println("✓ Loaded Flux")
flush(stdout)

using CUDA
println("✓ Loaded CUDA")
flush(stdout)

# Include the POMDP definition
include("pomdp_sequential_inference.jl")

# Global planner state
mutable struct PlannerState
    pomdp::Union{Nothing, CalculusTutoringPOMDP}
    planner::Union{Nothing, Any}
    updater::Union{Nothing, TutoringBeliefUpdater}
    current_belief::Union{Nothing, TutoringState}
    dialogue_transformer::Union{Nothing, SequentialDialogueTransformer}
    device::Union{Nothing, Function}
    problem_text::String
end

const SERVER_STATE = PlannerState(
    nothing, nothing, nothing, nothing,
    nothing, nothing, ""
)

function initialize_planner(
    dialogue_model_path::String="checkpoints/sequential_dialogue_final-v4_epoch_30.jld2",
    templates_path::String="templates.json",
    action_mapping_path::String="template_action_mapping.json",
    mathbert_url::String="http://localhost:8081"
)
    """Initialize the POMCPOW planner with sequential transformer and template cache"""
    println("\n" * "="^70)
    println("🚀 INITIALIZING CALCULUS POMCPOW PLANNER (TURN-LEVEL)")
    println("="^70)
    flush(stdout)

    # Load templates and action mappings
    println("\n📚 Step 1/5: Loading templates and action mappings...")
    flush(stdout)

    println("   Checking if templates file exists: $templates_path")
    flush(stdout)
    if isfile(templates_path)
        println("   Templates file found, parsing JSON...")
        flush(stdout)
        templates_data = JSON.parsefile(templates_path)
        println("   Loading templates from dict...")
        flush(stdout)
        load_templates_from_dict!(templates_data)
        println("   ✓ Templates loaded successfully")
        flush(stdout)
    else
        error("Templates file not found: $templates_path")
    end

    println("   Checking if action mapping file exists: $action_mapping_path")
    flush(stdout)
    if isfile(action_mapping_path)
        println("   Action mapping file found, parsing JSON...")
        flush(stdout)
        action_mapping_data = JSON.parsefile(action_mapping_path)
        println("   Loading template action mapping...")
        flush(stdout)
        load_template_action_mapping!(action_mapping_data)
        println("   ✓ Action mappings loaded successfully")
        flush(stdout)
    else
        error("Action mapping file not found: $action_mapping_path")
    end

    # Setup device
    println("\n🖥️  Step 2/5: Setting up device...")
    flush(stdout)
    println("   Checking CUDA availability...")
    flush(stdout)
    cuda_available = CUDA.functional()
    println("   CUDA available: $cuda_available")
    flush(stdout)
    SERVER_STATE.device = cuda_available ? gpu : cpu
    println("   Using: $(SERVER_STATE.device == gpu ? "GPU" : "CPU")")
    flush(stdout)

    # Load trained sequential dialogue transformer
    println("\n📦 Step 3/5: Loading sequential dialogue transformer...")
    flush(stdout)
    println("   Model path: $dialogue_model_path")
    flush(stdout)
    println("   Loading JLD2 file...")
    flush(stdout)
    model_data = JLD2.load(dialogue_model_path)
    println("   ✓ JLD2 file loaded")
    flush(stdout)
    println("   Extracting model from data...")
    flush(stdout)
    model_cpu = model_data["model"]
    println("   ✓ Model extracted")
    flush(stdout)

    println("   Moving model to device ($(SERVER_STATE.device == gpu ? "GPU" : "CPU"))...")
    flush(stdout)
    SERVER_STATE.dialogue_transformer = model_cpu |> SERVER_STATE.device
    println("   ✓ Model moved to device")
    flush(stdout)
    println("   Setting model to test mode...")
    flush(stdout)
    Flux.testmode!(SERVER_STATE.dialogue_transformer)
    println("   ✓ Model in test mode")
    flush(stdout)

    println("   ✓ Dialogue transformer loaded")
    flush(stdout)
    if haskey(model_data, "train_losses")
        println("   Final train loss: $(round(model_data["train_losses"][end], digits=4))")
        println("   Final val loss: $(round(model_data["val_losses"][end], digits=4))")
        flush(stdout)
    end

    # Load pre-computed template embedding cache from file (456 templates)
    println("\n🔮 Step 4/5: Loading pre-computed template embeddings cache...")
    flush(stdout)

    cache_file = joinpath(@__DIR__, "template_embeddings_cache.jld2")
    println("   Cache file: $cache_file")
    flush(stdout)

    try
        if isfile(cache_file)
            # Load the cache file
            cache_data = JLD2.load(cache_file)
            loaded_cache = cache_data["template_embeddings"]

            # Populate the global TEMPLATE_EMBEDDING_CACHE
            # Convert String keys to Int if needed
            # IMPORTANT: Move embeddings to GPU immediately if available
            for (template_id, embedding) in loaded_cache
                # template_id might be a String, convert to Int
                id_int = isa(template_id, String) ? parse(Int, template_id) : template_id
                # Move embedding to GPU once at load time, not on every use
                TEMPLATE_EMBEDDING_CACHE[id_int] = embedding |> SERVER_STATE.device
            end

            println("   ✓ Loaded $(length(TEMPLATE_EMBEDDING_CACHE)) template embeddings from cache")
            println("   ✓ All template embeddings moved to device ($(SERVER_STATE.device == gpu ? "GPU" : "CPU"))")
            flush(stdout)
        else
            println("   ⚠️  Cache file not found at $cache_file")
            println("   Templates will be encoded on-demand (will add latency)")
            flush(stdout)
        end
    catch e
        println("   ⚠️  Could not load template cache: $e")
        println("   Templates will be encoded on-demand (will add latency)")
        flush(stdout)
    end

    println("\n✅ Step 5/5: Planner initialization complete!")
    println("   • 245 calculus concepts")
    println("   • 456 templates (245 student, 211 tutor)")
    if length(TEMPLATE_EMBEDDING_CACHE) > 0
        println("   • $(length(TEMPLATE_EMBEDDING_CACHE)) template embeddings pre-cached")
    else
        println("   • Template embeddings will be cached on first use")
    end
    println("   • Delta-based belief updates")
    println("   • Turn-level sequential attention (not token-level)")
    println("   • MathBERT embeddings from TypeScript")
    println("="^70 * "\n")
    flush(stdout)
end

function create_pomdp_instance(problem_text::String)
    """Create POMDP instance for a specific problem"""
    println("📝 Creating POMDP for problem: \"$(first(problem_text, min(50, length(problem_text))))...\"")

    pomdp = CalculusTutoringPOMDP(
        dialogue_transformer = SERVER_STATE.dialogue_transformer,
        device = SERVER_STATE.device,
        problem_text = problem_text,
        initial_statement = problem_text,
        max_turns = 20,
        discount = 0.95f0,
        relevance_threshold = 0.3f0,
        alpha_belief_update = 0.7f0,
        w_relevance_reduction = 0.3f0,
        w_conversation_length = 0.2f0,
        w_coherence = 0.5f0
    )

    return pomdp
end

function handle_plan_request(req)
    """
    Handle /plan POST request (STATELESS)

    Expected request from Python/TypeScript:
    {
        "problem_text": "Find derivative of x²sin(x)",
        "dialogue_embedding": [0.23, -0.45, ...],  // 768-dim cumulative MathBERT embedding
        "turn_count": 3,                            // Number of real turns so far
        "x_features": [1.0, 0.0, ..., 8.5],        // 246 dims: speaker + 245 concepts
        "last_action": 3,                           // Last action taken (null/nothing on first turn)
        "last_tutor_action": 3                      // Last tutor action (null/nothing on first turn)
    }

    CRITICAL: Server is now STATELESS - all state passed in request
    """
    try
        # Parse request
        body = JSON.parse(req.body)
        problem_text = body["problem_text"]
        dialogue_embedding = Float32.(body["dialogue_embedding"])  # 768-dim cumulative embedding
        turn_count = Int(body["turn_count"])                      # Number of real turns
        x_features = Float32.(body["x_features"])

        # Parse last_action and last_tutor_action (can be null/nothing)
        last_action_raw = get(body, "last_action", nothing)
        last_tutor_action_raw = get(body, "last_tutor_action", nothing)

        # Convert to TutoringAction or nothing
        last_action = isnothing(last_action_raw) ? nothing : TutoringAction(Int(last_action_raw))
        last_tutor_action = isnothing(last_tutor_action_raw) ? nothing : TutoringAction(Int(last_tutor_action_raw))

        println("\n" * "="^70)
        println("📨 RECEIVED PLANNING REQUEST (STATELESS)")
        println("   Problem: $(first(problem_text, min(50, length(problem_text))))...")
        println("   Turn count: $(turn_count)")
        println("   Dialogue embedding dims: $(length(dialogue_embedding))")
        println("   Last action: $(last_action)")
        println("   Last tutor action: $(last_tutor_action)")

        # Extract from x_features: [speaker_indicator, relevance_245]
        speaker_indicator = x_features[1]
        current_relevance = x_features[2:end]  # 245 concepts

        println("   Speaker: $(speaker_indicator == 1.0 ? "Student" : "Tutor")")
        println("   Active concepts: $(sum(current_relevance .> 0.5))")
        println("   Concept mean: $(round(mean(current_relevance), digits=2))")

        # Create or update POMDP
        if SERVER_STATE.pomdp === nothing || SERVER_STATE.problem_text != problem_text
            println("\n🆕 Creating new POMDP instance")
            SERVER_STATE.pomdp = create_pomdp_instance(problem_text)
            SERVER_STATE.problem_text = problem_text

            # Create POMCPOW solver
            println("🔨 Creating POMCPOW solver...")
            solver = create_pomcpow_solver(
                SERVER_STATE.pomdp,
                n_iterations = 150,
                exploration_constant = 20.0f0,
                tree_depth = 4
            )

            # Solve to get planner/policy
            println("🎯 Solving POMDP to create planner...")
            SERVER_STATE.planner = solve(solver, SERVER_STATE.pomdp)

            # Create belief updater
            SERVER_STATE.updater = TutoringBeliefUpdater(SERVER_STATE.pomdp)

            println("✓ POMCPOW planner ready (200 simulations per action)")
        end

        # Create belief state (stateless - use actions from request)
        println("\n📍 Creating belief state from request (stateless)")
        current_belief = TutoringState(
            current_relevance,
            dialogue_embedding,  # Cumulative MathBERT embedding from request
            turn_count,          # Real turn count from request
            Int[],               # Empty simulated template history (for planning rollouts)
            Vector{Vector{Float32}}(),  # Empty simulated relevance history (for planning rollouts)
            last_action,         # From request (can be nothing on first turn)
            last_tutor_action    # From request (can be nothing on first turn)
        )


        # Plan action using POMCPOW (stateless - use local belief)
        action = POMDPs.action(SERVER_STATE.planner, current_belief)
        action_enum = Int(action)

        println("   POMCPOW selected action: $(ACTION_NAMES[action_enum]) (enum: $action_enum)")

        # Execute action to get observation with DELTA
        sp, o, r = execute_action_in_environment(
            SERVER_STATE.pomdp,
            current_belief,
            action
        )

        # Update belief for return (but don't store in SERVER_STATE - stateless!)
        updated_belief = POMDPs.update(
            SERVER_STATE.updater,
            current_belief,
            action,
            o
        )

        # Extract relevance DELTA from observation
        relevance_delta = o.relevance_delta  # This is the DELTA prediction

        # Prepare response
        response_data = Dict(
            "action" => action_enum,
            "relevance_delta" => Array{Float32}(relevance_delta),  # Return DELTA
            "reward" => r,
            "turn_count" => updated_belief.current_turn_count,
            "simulated_turns" => length(updated_belief.simulated_template_history),
            "speaker_next" => o.speaker_indicator,
            "template_distribution" => Array{Float32}(o.template_distribution)[1:min(20, end)]  # Top 20 for debugging
        )

        println("✅ Planning complete")
        println("="^70 * "\n")

        return HTTPResponse(200, JSON.json(response_data))

    catch e
        # Write to file immediately to bypass pipe buffering
        try
            error_log = open("/tmp/julia_errors.log", "a")
            println(error_log, "\n" * "="^70)
            println(error_log, "ERROR AT $(now())")
            println(error_log, "="^70)
            println(error_log, "\n❌ ERROR DURING PLANNING:")
            println(error_log, e)
            showerror(error_log, e, catch_backtrace())
            println(error_log, "\n" * "="^70 * "\n")
            close(error_log)
        catch file_err
            # If file logging fails, at least try stdout
            println("Failed to write to error log: $file_err")
        end

        # Also write to stdout (though it may not be visible)
        println("\n❌ ERROR DURING PLANNING:")
        println(e)
        showerror(stdout, e, catch_backtrace())
        println()
        flush(stdout)  # CRITICAL: Flush error output immediately

        error_response = Dict(
            "error" => string(e),
            "type" => string(typeof(e))
        )

        return HTTPResponse(500, JSON.json(error_response))
    end
end

function handle_reset_request(req)
    """Handle /reset POST request (server is stateless, so just clear caches)"""
    try
        println("\n" * "="^70)
        println("🔄 RESET REQUEST (STATELESS SERVER)")
        println("="^70)

        # Server is now stateless - no belief state to clear
        # Only clear problem-specific caches
        SERVER_STATE.problem_text = ""

        # Clear rollout cache if POMDP exists
        if SERVER_STATE.pomdp !== nothing
            cache_size = length(SERVER_STATE.pomdp.rollout_cache)
            empty!(SERVER_STATE.pomdp.rollout_cache)
            println("✓ Cleared rollout cache ($cache_size entries)")
        end

        println("✓ Reset complete (server is stateless, session state in Modal Dict)")
        println("="^70 * "\n")

        return HTTPResponse(200, JSON.json(Dict("status" => "reset")))

    catch e
        println("❌ ERROR during reset: $e")
        showerror(stdout, e, catch_backtrace())
        error_response = Dict("error" => string(e))
        return HTTPResponse(500, JSON.json(error_response))
    end
end

function handle_health_request(req)
    """Handle /health GET request"""
    status = Dict(
        "status" => "healthy",
        "domain" => "calculus",
        "num_concepts" => 245,
        "num_templates" => 456,
        "template_cache_size" => length(TEMPLATE_EMBEDDING_CACHE),
        "model_type" => "sequential_transformer_turn_level",
        "embedding_source" => "mathbert_cumulative_from_typescript",
        "device" => string(SERVER_STATE.device),
        "planner_initialized" => SERVER_STATE.planner !== nothing,
        "current_turn" => SERVER_STATE.current_belief !== nothing ?
                         SERVER_STATE.current_belief.current_turn_count : 0
    )
    return HTTPResponse(200, JSON.json(status))
end

# Router
function router(req)
    if req.method == "POST" && req.target == "/plan"
        return handle_plan_request(req)
    elseif req.method == "POST" && req.target == "/reset"
        return handle_reset_request(req)
    elseif req.method == "GET" && req.target == "/health"
        return handle_health_request(req)
    else
        return HTTPResponse(404, JSON.json(Dict("error" => "Not Found")))
    end
end

function main()
    """Start the HTTP server"""
    println("\n" * "="^70)
    println("🎓 POMCPOW CALCULUS TUTORING SERVER (TURN-LEVEL)")
    println("   Sequential Dialogue Transformer")
    println("   MathBERT Cumulative Embeddings from TypeScript")
    println("   Cached Template Embeddings for Planning")
    println("   Delta-Based Belief Updates")
    println("   245 Concepts | 456 Templates")
    println("="^70)
    flush(stdout)

    # Parse command line arguments
    println("\n🔧 Parsing command line arguments...")
    flush(stdout)
    port = 8080
    dialogue_model = "checkpoints/sequential_dialogue_final-v4_epoch_30.jld2"
    templates_path = "templates.json"
    action_mapping_path = "template_action_mapping.json"
    mathbert_url = "http://localhost:8081"

    if length(ARGS) >= 1
        port = parse(Int, ARGS[1])
    end
    if length(ARGS) >= 2
        dialogue_model = ARGS[2]
    end
    if length(ARGS) >= 3
        templates_path = ARGS[3]
    end
    if length(ARGS) >= 4
        action_mapping_path = ARGS[4]
    end
    if length(ARGS) >= 5
        mathbert_url = ARGS[5]
    end

    println("   Port: $port")
    println("   Dialogue model: $dialogue_model")
    println("   Templates: $templates_path")
    println("   Action mapping: $action_mapping_path")
    println("   MathBERT URL: $mathbert_url")
    flush(stdout)

    # Initialize planner (will pre-cache all 456 template embeddings)
    println("\n🚀 Calling initialize_planner()...")
    flush(stdout)
    initialize_planner(dialogue_model, templates_path, action_mapping_path, mathbert_url)
    println("✓ initialize_planner() completed successfully")
    flush(stdout)

    # Start server
    println("\n🌐 Starting HTTP server on port $port...")
    println("\nEndpoints:")
    println("  POST /plan   - Plan next tutoring action")
    println("                 Requires: dialogue_embedding (768-dim), turn_count, x_features")
    println("                 Returns: action, relevance_delta, reward")
    println("  POST /reset  - Reset session state")
    println("  GET  /health - Health check")
    println("\n✅ Server ready! Press Ctrl+C to stop.")
    println("   Architecture: Turn-level attention (O(turns²) not O(tokens²))")
    println("   Real dialogue: MathBERT cumulative embedding (from TypeScript)")
    println("   Simulations: Cached template embeddings (456 pre-computed)")
    println("="^70 * "\n")
    flush(stdout)

    # Use simple Sockets-based HTTP server (avoids HTTP.jl SSL issues)
    println("🔌 Creating SimpleHTTPServer instance...")
    flush(stdout)
    server = SimpleHTTPServer(port, router)
    println("✓ Server instance created")
    flush(stdout)

    println("🎧 Starting server.serve()...")
    flush(stdout)
    serve(server)
    println("⚠️  serve() returned (this should not happen unless server stopped)")
    flush(stdout)
end
# ACTION_NAMES is defined in pomdp_sequential_inference.jl (included above)

# Run server if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end