"""
HTTP Server for POMCPOW Calculus Tutoring Planner
Sequential Dialogue Transformer with Delta-Based Belief Updates
Uses MathBERT for first turn, lightweight encoder for planning simulations
"""

using HTTP
using JSON
using POMDPs
using POMCPOW
using Statistics
using JLD2
using Flux
using CUDA

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

    # Load templates and action mappings
    println("\n📚 Loading templates and action mappings...")

    if isfile(templates_path)
        templates_data = JSON.parsefile(templates_path)
        load_templates_from_dict!(templates_data)
    else
        error("Templates file not found: $templates_path")
    end

    if isfile(action_mapping_path)
        action_mapping_data = JSON.parsefile(action_mapping_path)
        load_template_action_mapping!(action_mapping_data)
    else
        error("Action mapping file not found: $action_mapping_path")
    end

    # Setup device
    println("\n🖥️  Setting up device...")
    println("   CUDA available: $(CUDA.functional())")
    SERVER_STATE.device = CUDA.functional() ? gpu : cpu
    println("   Using: $(SERVER_STATE.device == gpu ? "GPU" : "CPU")")

    # Load trained sequential dialogue transformer
    println("\n📦 Loading sequential dialogue transformer...")
    model_data = JLD2.load(dialogue_model_path)
    model_cpu = model_data["model"]

    SERVER_STATE.dialogue_transformer = model_cpu |> SERVER_STATE.device
    Flux.testmode!(SERVER_STATE.dialogue_transformer)

    println("   ✓ Dialogue transformer loaded")
    if haskey(model_data, "train_losses")
        println("   Final train loss: $(round(model_data["train_losses"][end], digits=4))")
        println("   Final val loss: $(round(model_data["val_losses"][end], digits=4))")
    end

    # Initialize template embedding cache (456 templates)
    println("\n🔮 Initializing template embedding cache with MathBERT...")
    println("   This will pre-compute embeddings for all 456 templates")
    println("   Estimated time: ~2 minutes")

    initialize_template_cache!(mathbert_url)

    println("\n✅ Planner initialization complete!")
    println("   • 245 calculus concepts")
    println("   • 456 templates (245 student, 211 tutor)")
    println("   • 456 template embeddings pre-cached")
    println("   • Delta-based belief updates")
    println("   • Turn-level sequential attention (not token-level)")
    println("   • MathBERT embeddings from TypeScript")
    println("="^70 * "\n")
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

function handle_plan_request(req::HTTP.Request)
    """
    Handle /plan POST request

    Expected request from TypeScript:
    {
        "problem_text": "Find derivative of x²sin(x)",
        "dialogue_embedding": [0.23, -0.45, ...],  // 768-dim cumulative MathBERT embedding
        "turn_count": 3,                            // Number of real turns so far
        "x_features": [1.0, 0.0, ..., 8.5]         // 246 dims: speaker + 245 concepts
    }

    CRITICAL: dialogue_embedding is the cumulative embedding of ALL dialogue so far
    """
    try
        # Parse request
        body = JSON.parse(String(req.body))
        problem_text = body["problem_text"]
        dialogue_embedding = Float32.(body["dialogue_embedding"])  # 768-dim cumulative embedding
        turn_count = Int(body["turn_count"])                      # Number of real turns
        x_features = Float32.(body["x_features"])

        println("\n" * "="^70)
        println("📨 RECEIVED PLANNING REQUEST (TURN-LEVEL)")
        println("   Problem: $(first(problem_text, min(50, length(problem_text))))...")
        println("   Turn count: $(turn_count)")
        println("   Dialogue embedding dims: $(length(dialogue_embedding))")

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
                n_iterations = 200,
                exploration_constant = 20.0f0,
                tree_depth = 5
            )

            # Solve to get planner/policy
            println("🎯 Solving POMDP to create planner...")
            SERVER_STATE.planner = solve(solver, SERVER_STATE.pomdp)

            # Create belief updater
            SERVER_STATE.updater = TutoringBeliefUpdater(SERVER_STATE.pomdp)

            println("✓ POMCPOW planner ready (200 simulations per action)")
        end

        # Create or update belief state
        if SERVER_STATE.current_belief === nothing || turn_count == 1
            println("\n📍 Initializing belief state")
            SERVER_STATE.current_belief = TutoringState(
                current_relevance,
                dialogue_embedding,  # Cumulative MathBERT embedding from TypeScript
                turn_count,          # Real turn count
                Int[],               # Empty simulated template history
                Vector{Vector{Float32}}(),  # Empty simulated relevance history
                nothing,             # last_action
                nothing              # last_tutor_action
            )
        else
            println("\n📍 Updating belief state (turn $(turn_count))")
            SERVER_STATE.current_belief = TutoringState(
                current_relevance,
                dialogue_embedding,  # Updated cumulative embedding from TypeScript
                turn_count,          # Updated turn count
                Int[],               # Clear simulated template history for new planning
                Vector{Vector{Float32}}(),  # Clear simulated relevance history for new planning
                SERVER_STATE.current_belief.last_action,
                SERVER_STATE.current_belief.last_tutor_action
            )
        end

        # 🐛 DEBUG: Log state being passed to POMCPOW
        println("\n🔍 DEBUG: State passed to POMCPOW:")
        println("   Relevance mean: $(round(mean(SERVER_STATE.current_belief.concept_relevance), digits=3))")
        println("   Relevance std: $(round(std(SERVER_STATE.current_belief.concept_relevance), digits=3))")
        println("   Relevance top 5 indices: ", sortperm(SERVER_STATE.current_belief.concept_relevance, rev=true)[1:5])
        println("   Relevance top 5 values: ", sort(SERVER_STATE.current_belief.concept_relevance, rev=true)[1:5])
        println("   Embedding mean: $(round(mean(SERVER_STATE.current_belief.current_dialogue_embedding), digits=3))")
        println("   Embedding std: $(round(std(SERVER_STATE.current_belief.current_dialogue_embedding), digits=3))")
        println("   Embedding first 5: ", SERVER_STATE.current_belief.current_dialogue_embedding[1:5])
        println("   Turn count: $(SERVER_STATE.current_belief.current_turn_count)")
        println("   Last action: $(SERVER_STATE.current_belief.last_action)")

        # Plan action using POMCPOW
        println("\n🤔 Running POMCPOW planning...")
        println("   Real dialogue: MathBERT cumulative embedding (from TypeScript)")
        println("   Simulated turns: Cached template embeddings (~200 sims)")

        # 🐛 DEBUG: Check what the transformer is predicting BEFORE POMCPOW
        speaker_pred, delta_pred, template_dist_pred = get_or_compute_forward_pass(
            SERVER_STATE.pomdp,
            SERVER_STATE.current_belief
        )

        # 🐛 DEBUG: Check turn-taking logic
        total_turns = SERVER_STATE.current_belief.current_turn_count +
                      length(SERVER_STATE.current_belief.simulated_template_history)
        println("\n🔍 DEBUG: Turn-taking logic:")
        println("   current_turn_count: $(SERVER_STATE.current_belief.current_turn_count)")
        println("   simulated_template_history length: $(length(SERVER_STATE.current_belief.simulated_template_history))")
        println("   total_turns: $total_turns")
        println("   total_turns % 2: $(total_turns % 2)")
        println("   Expected speaker: $(total_turns % 2 == 1 ? "TUTOR" : "STUDENT")")

        top_template_indices = sortperm(template_dist_pred, rev=true)[1:10]
        top_template_probs = softmax(template_dist_pred)[top_template_indices]
        println("\n🔍 DEBUG: Transformer predictions BEFORE masking:")
        println("   Top 10 template indices: $top_template_indices")
        println("   Top 10 template probs: $(round.(top_template_probs, digits=4))")

        # Apply masking manually to see what happens
        template_dist_masked = copy(template_dist_pred)
        if total_turns % 2 == 1
            # Odd turn: TUTOR speaks
            template_dist_masked[1:NUM_STUDENT_TEMPLATES] .= Float32(-1e9)
            println("   Masking: STUDENT templates (1-245) set to -1e9")
        else
            # Even turn: STUDENT speaks
            template_dist_masked[(NUM_STUDENT_TEMPLATES+1):end] .= Float32(-1e9)
            println("   Masking: TUTOR templates (246-456) set to -1e9")
        end

        top_template_indices_masked = sortperm(template_dist_masked, rev=true)[1:10]
        top_template_probs_masked = softmax(template_dist_masked)[top_template_indices_masked]
        println("\n🔍 DEBUG: After masking:")
        println("   Top 10 template indices: $top_template_indices_masked")
        println("   Top 10 template probs: $(round.(top_template_probs_masked, digits=4))")

        # Check action distribution from MASKED top templates
        println("\n   Action distributions for top 3 MASKED templates:")
        println("   Total templates with action mappings: $(length(TEMPLATE_ACTION_DISTRIBUTIONS))")

        for i in 1:min(3, length(top_template_indices_masked))
            tidx = top_template_indices_masked[i]
            if haskey(TEMPLATE_ACTION_DISTRIBUTIONS, tidx)
                action_probs = TEMPLATE_ACTION_DISTRIBUTIONS[tidx]
                top_action_idx = argmax(action_probs)
                println("      Template $tidx → $(ACTION_NAMES[top_action_idx]) (prob=$(round(action_probs[top_action_idx], digits=3)))")
                # Show all action probs
                println("         Full distribution: $(round.(action_probs, digits=3))")
            else
                println("      Template $tidx → MISSING FROM ACTION MAPPINGS!")
            end
        end

        action = POMDPs.action(SERVER_STATE.planner, SERVER_STATE.current_belief)
        action_enum = Int(action)

        println("\n   POMCPOW selected action: $(ACTION_NAMES[action_enum]) (enum: $action_enum)")

        # Execute action to get observation with DELTA
        sp, o, r = execute_action_in_environment(
            SERVER_STATE.pomdp,
            SERVER_STATE.current_belief,
            action
        )

        # Update belief for next turn
        SERVER_STATE.current_belief = POMDPs.update(
            SERVER_STATE.updater,
            SERVER_STATE.current_belief,
            action,
            o
        )

        # Extract relevance DELTA from observation
        relevance_delta = o.relevance_delta  # This is the DELTA prediction

        println("\n📊 Planning Results:")
        println("   Delta mean (abs): $(round(mean(abs.(relevance_delta)), digits=5))")
        println("   Delta max (abs): $(round(maximum(abs.(relevance_delta)), digits=5))")
        println("   Delta std: $(round(std(relevance_delta), digits=5))")
        println("   Non-zero deltas: $(sum(abs.(relevance_delta) .> 0.001))")
        println("   Top 5 positive deltas: ", sort(relevance_delta, rev=true)[1:5])
        println("   Top 5 negative deltas: ", sort(relevance_delta)[1:5])
        println("   Reward: $(round(r, digits=2))")
        println("   Turn count: $(SERVER_STATE.current_belief.current_turn_count)")
        println("   Simulated turns: $(length(SERVER_STATE.current_belief.simulated_template_history))")

        # Prepare response
        response_data = Dict(
            "action" => action_enum,
            "relevance_delta" => Array{Float32}(relevance_delta),  # Return DELTA
            "reward" => r,
            "turn_count" => SERVER_STATE.current_belief.current_turn_count,
            "simulated_turns" => length(SERVER_STATE.current_belief.simulated_template_history),
            "speaker_next" => o.speaker_indicator,
            "template_distribution" => Array{Float32}(o.template_distribution)[1:min(20, end)]  # Top 20 for debugging
        )

        println("✅ Planning complete")
        println("="^70 * "\n")

        return HTTP.Response(200, JSON.json(response_data))

    catch e
        println("\n❌ ERROR DURING PLANNING:")
        println(e)
        showerror(stdout, e, catch_backtrace())
        println()

        error_response = Dict(
            "error" => string(e),
            "type" => string(typeof(e))
        )

        return HTTP.Response(500, JSON.json(error_response))
    end
end

function handle_reset_request(req::HTTP.Request)
    """Handle /reset POST request to start new session"""
    try
        println("\n" * "="^70)
        println("🔄 RESETTING PLANNER STATE")
        println("="^70)

        # Clear current belief and problem
        SERVER_STATE.current_belief = nothing
        SERVER_STATE.problem_text = ""

        # Clear rollout cache if POMDP exists
        if SERVER_STATE.pomdp !== nothing
            cache_size = length(SERVER_STATE.pomdp.rollout_cache)
            empty!(SERVER_STATE.pomdp.rollout_cache)
            println("✓ Cleared rollout cache ($cache_size entries)")
        end

        println("✓ Reset complete - ready for new session")
        println("="^70 * "\n")

        return HTTP.Response(200, JSON.json(Dict("status" => "reset")))

    catch e
        println("❌ ERROR during reset: $e")
        showerror(stdout, e, catch_backtrace())
        error_response = Dict("error" => string(e))
        return HTTP.Response(500, JSON.json(error_response))
    end
end

function handle_health_request(req::HTTP.Request)
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
    return HTTP.Response(200, JSON.json(status))
end

# Router
function router(req::HTTP.Request)
    if req.method == "POST" && req.target == "/plan"
        return handle_plan_request(req)
    elseif req.method == "POST" && req.target == "/reset"
        return handle_reset_request(req)
    elseif req.method == "GET" && req.target == "/health"
        return handle_health_request(req)
    else
        return HTTP.Response(404, JSON.json(Dict("error" => "Not Found")))
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

    # Parse command line arguments
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

    # Initialize planner (will pre-cache all 456 template embeddings)
    initialize_planner(dialogue_model, templates_path, action_mapping_path, mathbert_url)

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

    HTTP.serve(router, "0.0.0.0", port)
end

# Action name mapping for logging (CORRECTED to match enum)
const ACTION_NAMES = Dict(
    1 => "instruct",
    2 => "connect",
    3 => "correct",
    4 => "probe",
    5 => "guide",
    6 => "affirm",
    7 => "listen",
    8 => "practice"
)

# Run server if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end