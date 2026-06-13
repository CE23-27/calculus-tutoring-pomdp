"""
POMDP Definition for Calculus Tutoring with Sequential Transformer
Complete implementation with solver configuration
Save as: pomcpow_calculus_tutoring.jl
"""

using POMDPs
using POMDPTools
using POMCPOW
using BasicPOMCP
using Flux
using Statistics
using Random
using LinearAlgebra
using Distributions
using StatsBase
using HTTP
using JSON3

# ========================
# Constants
# ========================
const NUM_CONCEPTS = 245
const NUM_TEMPLATES = 456
const NUM_STUDENT_TEMPLATES = 245  # First 245 templates
const NUM_TUTOR_TEMPLATES = 211    # Last 211 templates

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
# ========================
# Global Dictionaries (loaded from data)
# ========================
const TEMPLATE_TEXTS = Dict{Int, String}()
const TEMPLATE_ACTION_DISTRIBUTIONS = Dict{Int, Vector{Float32}}()
const ACTION_NAME_TO_IDX = Dict(v => k for (k, v) in ACTION_NAMES)
# This creates: "instruct" => 1, "connect" => 2, "correct" => 3, etc.

# Global template embedding cache (computed at startup with MathBERT)
# Note: Values can be either Vector{Float32} (CPU) or CuArray{Float32} (GPU)
const TEMPLATE_EMBEDDING_CACHE = Dict{Int, Any}()

function load_templates_from_dict!(templates_dict::Dict)
    """
    Load template texts from dictionary
    Expected format: Dict(template_idx => template_text)
    """
    empty!(TEMPLATE_TEXTS)
    for (idx, text) in templates_dict
        TEMPLATE_TEXTS[parse(Int, idx)] = String(text)  # ✅ Changed Int(idx) to parse(Int, idx)
    end
    println("✓ Loaded $(length(TEMPLATE_TEXTS)) template texts")
end

function load_template_action_mapping!(mapping_dict::Dict)
    """
    Load template→action distribution mapping
    Expected format: Dict(template_idx => [count_instruct, count_connect, ..., count_practice])
    Converts counts to probability distributions
    Each template maps to 8-dimensional probability vector over actions
        """
    empty!(TEMPLATE_ACTION_DISTRIBUTIONS)
    for (template_idx, action_counts) in mapping_dict
        # Create array for 8 actions
        counts = zeros(Float32, 8)

        # Map action names to indices using your ACTION_NAMES mapping
        for (action_name, count) in action_counts
            if haskey(ACTION_NAME_TO_IDX, action_name)
                action_idx = ACTION_NAME_TO_IDX[action_name]
                counts[action_idx] = Float32(count)
            else
                @warn "Unknown action name: $action_name"
            end
        end

        # Normalize to probability distribution
        total = sum(counts)
        if total > 0.0f0
            probs = counts ./ total
        else
            @warn "Template $template_idx has no action counts, using uniform distribution"
            probs = ones(Float32, 8) ./ 8.0f0
        end

        # Verify it's a valid probability distribution
        if !isapprox(sum(probs), 1.0f0, atol=0.001f0)
            @warn "Template $template_idx probabilities don't sum to 1.0 (sum=$(sum(probs))), renormalizing..."
            probs = probs ./ sum(probs)
        end

        TEMPLATE_ACTION_DISTRIBUTIONS[parse(Int, template_idx)] = probs
    end
    println("✓ Loaded and normalized $(length(TEMPLATE_ACTION_DISTRIBUTIONS)) template-action mappings")
end

# ========================
# MathBERT Client
# ========================

struct MathBERTClient
    server_url::String
end

function encode_text_mathbert(client::MathBERTClient, text::String)
    """
    Send text to MathBERT server and get 768-dimensional embedding
    Used for pre-computing template embeddings at startup
    """
    try
        response = HTTP.post(
            "$(client.server_url)/embed",
            ["Content-Type" => "application/json"],
            JSON3.write(Dict("text" => text))
        )

        result = JSON3.read(String(response.body))
        return Float32.(result.embedding)
    catch e
        println("Error calling MathBERT server: $e")
        rethrow(e)
    end
end

function initialize_template_cache!(mathbert_url::String="http://localhost:8081")
    """
    Pre-compute embeddings for all 456 templates using MathBERT
    This is called once at server startup
    Estimated time: ~2 minutes for 456 templates
    """
    println("🔮 Initializing template embedding cache with MathBERT...")
    println("   This will take approximately 2 minutes...")

    empty!(TEMPLATE_EMBEDDING_CACHE)
    client = MathBERTClient(mathbert_url)

    total = length(TEMPLATE_TEXTS)
    for (idx, (template_idx, template_text)) in enumerate(TEMPLATE_TEXTS)
        if idx % 50 == 0
            println("   Progress: $idx/$total templates encoded")
        end

        embedding = encode_text_mathbert(client, template_text)
        TEMPLATE_EMBEDDING_CACHE[template_idx] = embedding
    end

    println("✓ Template cache initialized: $(length(TEMPLATE_EMBEDDING_CACHE)) templates")
    println("   Each template has $(length(first(values(TEMPLATE_EMBEDDING_CACHE)))) dimensions")
end 

# ========================
# Transformer Components (for forward passes)
# ========================

struct DialogueMultiHeadAttention
    n_heads::Int
    d_model::Int
    d_k::Int
    W_q::Dense
    W_k::Dense
    W_v::Dense
    W_o::Dense
    dropout::Dropout
end

Flux.@layer DialogueMultiHeadAttention

function DialogueMultiHeadAttention(d_model::Int, n_heads::Int; dropout_rate=Float32(0.1))
    @assert d_model % n_heads == 0
    d_k = d_model ÷ n_heads
    
    DialogueMultiHeadAttention(
        n_heads,
        d_model,
        d_k,
        Dense(d_model, d_model, bias=false),
        Dense(d_model, d_model, bias=false),
        Dense(d_model, d_model, bias=false),
        Dense(d_model, d_model),
        Dropout(dropout_rate)
    )
end

function (mha::DialogueMultiHeadAttention)(x::AbstractArray, mask=nothing)
    hidden_dim, seq_len, batch_size = size(x)
    
    Q = mha.W_q(x)
    K = mha.W_k(x)
    V = mha.W_v(x)
    
    Q = reshape(Q, mha.d_k, mha.n_heads, seq_len, batch_size)
    K = reshape(K, mha.d_k, mha.n_heads, seq_len, batch_size)
    V = reshape(V, mha.d_k, mha.n_heads, seq_len, batch_size)
    
    Q = permutedims(Q, (1, 3, 2, 4))
    K = permutedims(K, (1, 3, 2, 4))
    V = permutedims(V, (1, 3, 2, 4))
    
    Q = reshape(Q, mha.d_k, seq_len, mha.n_heads * batch_size)
    K = reshape(K, mha.d_k, seq_len, mha.n_heads * batch_size)
    V = reshape(V, mha.d_k, seq_len, mha.n_heads * batch_size)
    
    scores = batched_mul(
        permutedims(Q, (2, 1, 3)),
        permutedims(K, (1, 2, 3))
    ) / sqrt(Float32(mha.d_k))
    
    if mask !== nothing
        scores = scores .+ (mask .* Float32(-1e9))
    end
    
    attn_weights = softmax(scores; dims=2)
    attn_weights = mha.dropout(attn_weights)
    
    output = batched_mul(attn_weights, permutedims(V, (2, 1, 3)))
    
    output = permutedims(output, (2, 1, 3))
    output = reshape(output, mha.d_k, seq_len, mha.n_heads, batch_size)
    output = permutedims(output, (1, 3, 2, 4))
    output = reshape(output, hidden_dim, seq_len, batch_size)
    
    return mha.W_o(output)
end

struct TransformerBlock
    attention::DialogueMultiHeadAttention
    norm1::LayerNorm
    ffn::Chain
    norm2::LayerNorm
    dropout::Dropout
end

Flux.@layer TransformerBlock

function TransformerBlock(d_model::Int, n_heads::Int, ff_dim::Int; dropout_rate=Float32(0.1))
    TransformerBlock(
        DialogueMultiHeadAttention(d_model, n_heads; dropout_rate=dropout_rate),
        LayerNorm(d_model),
        Chain(
            Dense(d_model, ff_dim, relu),
            Dropout(dropout_rate),
            Dense(ff_dim, d_model)
        ),
        LayerNorm(d_model),
        Dropout(dropout_rate)
    )
end

function (tb::TransformerBlock)(x::AbstractArray, mask=nothing)
    attn_output = tb.attention(x, mask)
    x = x .+ tb.dropout(attn_output)
    x = tb.norm1(x)
    
    ffn_output = tb.ffn(x)
    x = x .+ tb.dropout(ffn_output)
    x = tb.norm2(x)
    
    return x
end

struct SequentialDialogueTransformer
    turn_projection::Dense
    positional_encoding::Array{Float32, 2}
    transformer_blocks::Vector{TransformerBlock}
    output_projection::Dense
    dropout::Dropout
end

Flux.@layer SequentialDialogueTransformer trainable=(turn_projection, transformer_blocks, output_projection, dropout)

function SequentialDialogueTransformer(
    input_dim::Int,
    hidden_dim::Int,
    n_heads::Int,
    n_layers::Int,
    ff_dim::Int,
    output_dim::Int,
    max_seq_length::Int;
    dropout_rate=Float32(0.1)
)
    # Create positional encoding
    pos_enc = create_positional_encoding(hidden_dim, max_seq_length)
    
    SequentialDialogueTransformer(
        Dense(input_dim, hidden_dim),
        pos_enc,
        [TransformerBlock(hidden_dim, n_heads, ff_dim; dropout_rate=dropout_rate) for _ in 1:n_layers],
        Dense(hidden_dim, output_dim),
        Dropout(dropout_rate)
    )
end

function create_positional_encoding(d_model::Int, max_len::Int)
    """Create sinusoidal positional encoding"""
    pe = zeros(Float32, d_model, max_len)
    
    position = Float32.(0:max_len-1)'  # [1, max_len]
    div_term = exp.(Float32.(0:2:d_model-1) .* -(log(Float32(10000.0)) / d_model))  # [d_model/2]
    
    pe[1:2:end, :] .= sin.(position .* div_term)
    pe[2:2:end, :] .= cos.(position .* div_term)
    
    return pe
end

function create_causal_mask(seq_len::Int)
    mask = triu(ones(Float32, seq_len, seq_len), 1)
    return mask
end

# ========================
# POMDP Structures
# ========================


@enum TutoringAction instruct=1 connect=2 correct=3 probe=4 guide=5 affirm=6 listen=7 practice=8

# Action-to-student-template mapping
# Maps each tutor action to valid student template indices (1-245)
# Student templates that are reasonable responses to each pedagogical action
const ACTION_STUDENT_TEMPLATE_FILTERS = Dict{TutoringAction, Vector{Int}}(
    instruct => [1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 20, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 40, 44, 48, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60, 63, 64, 65, 66, 67, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 128, 130, 131, 132, 133, 134, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245],
    probe => [3, 17, 18, 22, 35, 39, 41, 42, 43, 45, 46, 47, 52, 61, 62, 68, 75, 127, 129, 135],
    guide => [1, 2, 6, 7, 8, 11, 12, 21, 23, 34, 36, 37, 38, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60, 64, 65, 69, 71, 72, 73, 74, 80, 81, 82, 84, 85, 88, 89, 90, 92, 94, 96, 97, 98, 99, 100, 101, 104, 107, 110, 111, 113, 114, 119, 120, 123, 128, 130, 131, 133, 134, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 162, 163, 165, 166, 167, 168, 169, 170, 171, 172, 173, 175, 176, 179, 181, 182, 183, 184, 185, 191, 193, 194, 195, 196, 197, 198, 199, 200, 201, 203, 204, 205, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 218, 220, 222, 224, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244],
    connect => [5, 9, 10, 13, 14, 15, 16, 17, 19, 20, 25, 27, 31, 32, 41, 42, 46, 47, 70, 71, 78, 79, 82, 83, 85, 87, 91, 93, 95, 99, 102, 103, 105, 108, 112, 115, 117, 118, 124, 125, 126, 127, 132, 136, 137, 139, 140, 142, 144, 146, 149, 164, 171, 174, 178, 192, 202, 220, 221, 225],
    correct => [16, 31, 52, 61, 62, 75, 127, 129, 135],
    affirm => [6, 7, 8, 21, 36, 40, 44, 48],
    listen => [3, 17, 18, 22, 35, 39, 41, 43, 45, 46, 52, 61, 62, 68, 75],
    practice => [33, 48, 72, 74, 104, 107, 180, 184, 189, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 206]
)

# Good pedagogical action sequences that should be rewarded
const GOOD_ACTION_PAIRS = Set([
    # Discovery/Assessment → Scaffolding
    (probe, guide),
    (probe, connect),
    (listen, guide),

    # Teaching → Assessment
    (instruct, probe),
    (connect, probe),
    (guide, probe),

    # Error Correction Flow
    (guide, correct),
    (listen, correct),
    (correct, guide)
])

struct TutoringState
    concept_relevance::Vector{Float32}  # Current relevance (245 concepts)
    current_dialogue_embedding::Vector{Float32}  # 768-dim cumulative MathBERT embedding of full conversation
    current_turn_count::Int  # Number of real dialogue turns so far
    simulated_template_history::Vector{Int}  # Future simulated turns (for planning only)
    simulated_relevance_history::Vector{Vector{Float32}}  # Relevance at each simulated turn (parallel to template_history)
    last_action::Union{Nothing, TutoringAction}  # Last action (any turn)
    last_tutor_action::Union{Nothing, TutoringAction}  # Last tutor action (for coherence bonus)
end

struct TutoringObservation
    speaker_indicator::Float32
    relevance_delta::Vector{Float32}
    template_distribution::Vector{Float32}
end

# POMCPOW needs to be able to sample from the belief state (deterministic)
Base.rand(rng::AbstractRNG, b::TutoringState) = b
# ========================

struct CalculusTutoringPOMDP <: POMDP{TutoringState, TutoringAction, TutoringObservation}
    dialogue_transformer::SequentialDialogueTransformer
    device::Function
    problem_text::String
    initial_statement::String
    max_turns::Int
    discount::Float32
    relevance_threshold::Float32
    alpha_belief_update::Float32
    w_relevance_reduction::Float32
    w_conversation_length::Float32
    w_coherence::Float32
    rollout_cache::Dict{Any,Any}
end

function CalculusTutoringPOMDP(;
    dialogue_transformer::SequentialDialogueTransformer,
    device::Function,
    problem_text::String,
    initial_statement::String,
    max_turns::Int=20,
    discount::Float32=0.95f0,
    relevance_threshold::Float32=0.3f0,
    alpha_belief_update::Float32=0.7f0,
    w_relevance_reduction::Float32=0.3f0,
    w_conversation_length::Float32=0.2f0,
    w_coherence::Float32=0.5f0
)
    return CalculusTutoringPOMDP(
        dialogue_transformer,
        device,
        problem_text,
        initial_statement,
        max_turns,
        discount,
        relevance_threshold,
        alpha_belief_update,
        w_relevance_reduction,
        w_conversation_length,
        w_coherence,
        Dict{Any,Any}()
    )
end

# ========================
# POMDP Interface
# ========================

POMDPs.states(pomdp::CalculusTutoringPOMDP) = nothing
POMDPs.actions(pomdp::CalculusTutoringPOMDP) = instances(TutoringAction)
POMDPs.observations(pomdp::CalculusTutoringPOMDP) = nothing
POMDPs.discount(pomdp::CalculusTutoringPOMDP) = pomdp.discount
POMDPs.isterminal(pomdp::CalculusTutoringPOMDP, s::TutoringState) =
    (s.current_turn_count + length(s.simulated_template_history)) >= pomdp.max_turns

# ========================
# Transformer Forward Pass
# ========================

function run_transformer_forward_pass(
    pomdp::CalculusTutoringPOMDP,
    state::TutoringState
)
    """
    Run full forward pass through dialogue transformer
    Uses pre-cached template embeddings for simulated turns
    Returns: (speaker_indicator, concept_delta, template_distribution)
    """

    global simulation_count = get(task_local_storage(), :sim_count, 0) + 1
    task_local_storage(:sim_count, simulation_count)
    if simulation_count % 100 == 0
        println("   Simulation #$simulation_count")
    end

    # Build embedding sequence: current dialogue embedding + cached template embeddings
    embeddings = []

    # First element: current real dialogue embedding from MathBERT (sent by TypeScript)
    push!(embeddings, state.current_dialogue_embedding |> pomdp.device)

    # Append cached template embeddings for simulated turns
    for template_idx in state.simulated_template_history
        if haskey(TEMPLATE_EMBEDDING_CACHE, template_idx)
            push!(embeddings, TEMPLATE_EMBEDDING_CACHE[template_idx] |> pomdp.device)
        else
            @warn "Template $template_idx not in cache, using zero embedding"
            push!(embeddings, zeros(Float32, 768) |> pomdp.device)
        end
    end

    # Total turns: 1 (real dialogue) + num_simulated
    n_turns = 1 + length(state.simulated_template_history)
    text_encodings = hcat(embeddings...)  # [768, n_turns]

    # Build turn features (concepts and speaker indicators)
    turn_concept_vectors = []
    turn_speakers = Float32[]

    # First turn: use current relevance (from the real dialogue state)
    push!(turn_concept_vectors, state.concept_relevance |> pomdp.device)
    push!(turn_speakers, Float32(0.0))  # Will be updated to correct speaker

    # Subsequent turns: use historical relevance from simulated_relevance_history
    for i in 1:length(state.simulated_relevance_history)
        push!(turn_concept_vectors, state.simulated_relevance_history[i] |> pomdp.device)
        push!(turn_speakers, Float32(0.0))  # Will be updated to correct speaker
    end

    # Combine into turn matrix: [768 + 1 + 245, n_turns] = [1014, n_turns]
    turn_matrix = vcat(
        text_encodings,
        reshape(turn_speakers, 1, :) |> pomdp.device,
        hcat(turn_concept_vectors...)
    )

    # Forward through transformer
    model = pomdp.dialogue_transformer

    hidden = model.turn_projection(turn_matrix)
    pos_enc_slice = model.positional_encoding[:, 1:n_turns] |> pomdp.device
    hidden = hidden .+ pos_enc_slice
    hidden = reshape(hidden, size(hidden, 1), size(hidden, 2), 1)

    causal_mask = create_causal_mask(n_turns) |> pomdp.device

    for block in model.transformer_blocks
        hidden = block(hidden, causal_mask)
    end

    last_token = hidden[:, end, 1]
    last_token = model.dropout(last_token)
    pred = model.output_projection(last_token)

    # Move to CPU and extract predictions
    pred_cpu = pred |> cpu

    speaker_pred = pred_cpu[1]
    concept_delta_pred = pred_cpu[2:246]
    template_dist_pred = pred_cpu[247:end]

    return (
        Float32(speaker_pred),
        Array{Float32}(concept_delta_pred),
        Array{Float32}(template_dist_pred)
    )
end
# ========================
# Get Single forward Pass
# ========================

function get_or_compute_forward_pass(pomdp::CalculusTutoringPOMDP, state::TutoringState)
    """Compute forward pass once per unique state"""

    # Cache key based on:
    # 1. Current concept relevance (the state we're predicting from)
    # 2. Dialogue embedding (the conversation context)
    # 3. Simulated template history (what templates were used in simulation)
    # 4. Simulated relevance history (how relevance evolved during simulation)
    # All 4 are needed to uniquely identify the state

    cache_key = hash((
        state.concept_relevance,
        state.current_dialogue_embedding,
        state.simulated_template_history,
        state.simulated_relevance_history
    ))

    if haskey(pomdp.rollout_cache, cache_key)
        return pomdp.rollout_cache[cache_key]
    else
        result = run_transformer_forward_pass(pomdp, state)
        pomdp.rollout_cache[cache_key] = result
        return result
    end
end
# ========================
# OBSERVATION MODEL
# ========================

function POMDPs.observation(pomdp::CalculusTutoringPOMDP, 
                            a::TutoringAction, 
                            sp::TutoringState)
    return TutoringObservationDistribution(pomdp, sp)
end

struct TutoringObservationDistribution
    pomdp::CalculusTutoringPOMDP
    state::TutoringState
end

function Base.rand(rng::AbstractRNG, d::TutoringObservationDistribution)
    """Sample observation by running transformer forward pass"""
    speaker, delta, template_dist = get_or_compute_forward_pass(d.pomdp, d.state)
    return TutoringObservation(speaker, delta, template_dist)
end

function Distributions.pdf(d::TutoringObservationDistribution, o::TutoringObservation)
    """Observation likelihood (deterministic model)"""
    return 1.0
end

# ========================
# TRANSITION MODEL
# ========================

function POMDPs.transition(pomdp::CalculusTutoringPOMDP, 
                           s::TutoringState, 
                           a::TutoringAction)
    return TutoringTransitionDistribution(pomdp, s, a)
end

struct TutoringTransitionDistribution
    pomdp::CalculusTutoringPOMDP
    current_state::TutoringState
    action::TutoringAction
end

function Base.rand(rng::AbstractRNG, d::TutoringTransitionDistribution)
    """
    Sample next state during POMCPOW simulation:
    1. Run transformer on [real_history + simulated_history]
    2. Mask templates based on last action
    3. Sample template index
    4. Append to simulated_template_history (fast!)
    5. Update relevance using delta
    
    Note: Real dialogue history stays unchanged during simulations
    """
    
    # Get transformer prediction
    speaker, delta, template_dist = get_or_compute_forward_pass(d.pomdp, d.current_state)

    # Deterministic turn-taking: student provides problem (turn 0), then tutor responds, then alternates
    # Total turns = real turns + simulated turns
    # Turn 1 (odd): TUTOR responds to problem
    # Turn 2 (even): STUDENT responds
    # Turn 3 (odd): TUTOR responds, etc.
    total_turns = d.current_state.current_turn_count + length(d.current_state.simulated_template_history)

    template_dist_masked = copy(template_dist)
    actual_action = d.action  # Default: use the action passed in

    if total_turns % 2 == 1
        # Odd turn: TUTOR speaks
        # The action d.action represents the tutor's pedagogical move
        template_dist_masked[1:NUM_STUDENT_TEMPLATES] .= Float32(-1e9)  # Mask student templates
    else
        # Even turn: STUDENT speaks (responding to previous tutor action)
        # Action for this turn is "listen" (tutor is listening to student)
        actual_action = listen
        template_dist_masked[(NUM_STUDENT_TEMPLATES+1):end] .= Float32(-1e9)  # Mask tutor templates

        # Filter student templates based on PREVIOUS tutor action
        if d.current_state.last_action !== nothing
            valid_templates = get(ACTION_STUDENT_TEMPLATE_FILTERS,
                                 d.current_state.last_action,
                                 1:NUM_STUDENT_TEMPLATES)

            # Create action filter mask: zero out invalid templates
            action_mask = zeros(Float32, length(template_dist_masked))
            for idx in valid_templates
                if idx <= NUM_STUDENT_TEMPLATES
                    action_mask[idx] = 1.0f0
                end
            end

            # Apply action filter (multiplicative - keeps speaker masking)
            template_dist_masked = template_dist_masked .+ (1.0f0 .- action_mask) .* Float32(-1e9)
        end
    end

    # Sample template
    template_probs = softmax(template_dist_masked)
    template_idx = sample(rng, 1:length(template_probs), Weights(template_probs))

    # Update relevance with smooth blending (not addition)
    # Step 1: Clamp predictions to non-negative (relevance is strictly [0, 10])
    delta_positive = max.(0.0f0, delta)

    # Step 2: Threshold to match training sparsity (zero out weak predictions)
    threshold = 2.0f0
    delta_thresholded = (delta_positive .>= threshold) .* delta_positive

    # Step 3: Smooth blend toward prediction (not accumulation)
    alpha = d.pomdp.alpha_belief_update  # Default 0.7
    new_relevance = alpha .* delta_thresholded .+ (1.0f0 .- alpha) .* d.current_state.concept_relevance
    new_relevance = clamp.(new_relevance, 0.0f0, 10.0f0)

    # Append to simulated history (FAST - no string operations!)
    # Real dialogue history is unchanged during simulations
    new_simulated_template_history = vcat(d.current_state.simulated_template_history, [template_idx])

    # Append NEW relevance to relevance history
    # This creates the parallel history of relevance states corresponding to each template
    new_simulated_relevance_history = vcat(d.current_state.simulated_relevance_history, [new_relevance])

    # Update last_tutor_action only on tutor turns (odd turns)
    new_last_tutor_action = d.current_state.last_tutor_action
    if total_turns % 2 == 1
        # Tutor turn: update last_tutor_action to the current action
        new_last_tutor_action = d.action
    end
    # On student turns (even), keep the previous last_tutor_action unchanged

    return TutoringState(
        new_relevance,  # Current relevance (after applying delta)
        d.current_state.current_dialogue_embedding,  # Keep dialogue embedding unchanged
        d.current_state.current_turn_count,  # Keep turn count unchanged during simulation
        new_simulated_template_history,  # Append new template
        new_simulated_relevance_history,  # Append new relevance state
        actual_action,  # Last action taken (d.action for tutor turns, listen for student turns)
        new_last_tutor_action  # Last tutor action (for coherence bonus)
    )
end

function Distributions.pdf(d::TutoringTransitionDistribution, sp::TutoringState)
    """Transition likelihood"""
    return 1.0
end

# ========================
# REWARD MODEL
# ========================

function POMDPs.reward(pomdp::CalculusTutoringPOMDP, 
                       s::TutoringState, 
                       a::TutoringAction, 
                       sp::TutoringState)
    """
    Compute reward based on:
    1. Positive relevance reduction (concepts becoming less relevant)
    2. Conversation length penalty (total real + simulated turns)
    3. Action coherence bonus
    """
    
    # Component 1: Positive relevance reduction
    old_high = sum(s.concept_relevance .> pomdp.relevance_threshold)
    new_high = sum(sp.concept_relevance .> pomdp.relevance_threshold)
    relevance_reduction = old_high - new_high
    
    # Only reward POSITIVE reductions
    relevance_reward = pomdp.w_relevance_reduction * max(0.0f0, relevance_reduction)
    
    # Component 2: Length penalty (real + simulated turns)
    total_turns = sp.current_turn_count + length(sp.simulated_template_history)
    turn_penalty = -pomdp.w_conversation_length * total_turns

    # Component 3: Coherence bonus for good pedagogical action sequences
    coherence_reward = 0.0f0
    if sp.last_tutor_action !== nothing
        action_pair = (sp.last_tutor_action, a)
        if action_pair in GOOD_ACTION_PAIRS
            coherence_reward = pomdp.w_coherence
        end
    end

    return Float32(relevance_reward + turn_penalty + coherence_reward)
end

# ========================
# ACTION SELECTION FROM TEMPLATE
# ========================

function select_action_from_template(template_idx::Int, rng::AbstractRNG)
    """
    Sample action from template's action distribution
    Restores stochasticity in action selection
    """
    if !haskey(TEMPLATE_ACTION_DISTRIBUTIONS, template_idx)
        # Uniform fallback
        actions = instances(TutoringAction)
        selected = rand(rng, actions)
        # 🐛 DEBUG
        println("      🎲 Template $template_idx MISSING → Uniform random → $(ACTION_NAMES[Int(selected)])")
        return selected
    end

    action_dist = TEMPLATE_ACTION_DISTRIBUTIONS[template_idx]
    actions = [instruct, connect, correct, probe, guide, affirm, listen, practice]

    r = rand(rng, Float32)
    cumsum = 0.0f0

    for (i, prob) in enumerate(action_dist)
        cumsum += prob
        if r <= cumsum
            # 🐛 DEBUG (only print occasionally to avoid spam)
            if rand(rng) < 0.01  # 1% sampling
                println("      🎲 Template $template_idx → $(ACTION_NAMES[i]) (r=$r, cumsum=$cumsum)")
            end
            return actions[i]
        end
    end

    # 🐛 DEBUG
    println("      ⚠️  Template $template_idx → Fallback to listen (r=$r exceeded all cumsum)")
    return listen
end

# ========================
# BELIEF UPDATER
# ========================

struct TutoringBeliefUpdater <: Updater
    pomdp::CalculusTutoringPOMDP
end

function POMDPs.update(
    up::TutoringBeliefUpdater,
    b::TutoringState,
    a::TutoringAction,
    o::TutoringObservation
)
    """
    Update belief after taking action and receiving observation

    NOTE: This is called after executing an action in the REAL environment.
    The dialogue embedding and turn count will be updated by TypeScript when it calls
    the server again for the next turn. Here we just update relevance and clear simulated history.
    """

    # Step 1: Clamp predictions to non-negative (relevance is strictly [0, 10])
    delta_positive = max.(0.0f0, o.relevance_delta)

    # Step 2: Threshold to match training sparsity (zero out weak predictions)
    threshold = 2.0f0
    delta_thresholded = (delta_positive .>= threshold) .* delta_positive

    # Step 3: Smooth blend toward prediction (not accumulation)
    alpha = up.pomdp.alpha_belief_update  # Default 0.7
    new_relevance = alpha .* delta_thresholded .+ (1.0f0 .- alpha) .* b.concept_relevance
    new_relevance = clamp.(new_relevance, 0.0f0, 10.0f0)

    # Increment turn count (we just took a real turn)
    new_turn_count = b.current_turn_count + 1

    # Update last_tutor_action if this is a tutor turn
    # In real execution, action 'a' is always from tutor (TypeScript waits for tutor response)
    new_last_tutor_action = (a != listen) ? a : b.last_tutor_action

    # Dialogue embedding stays the same - it will be updated by TypeScript on next request
    # Clear simulated history (we executed an action, simulations are done)
    new_simulated_history = Int[]
    new_simulated_relevance_history = Vector{Vector{Float32}}()

    return TutoringState(new_relevance, b.current_dialogue_embedding, new_turn_count, new_simulated_history, new_simulated_relevance_history, a, new_last_tutor_action)
end

# ========================
# ROLLOUT POLICY
# ========================

struct TutoringRolloutPolicy <: Policy
    pomdp::CalculusTutoringPOMDP
end

function POMDPs.action(policy::TutoringRolloutPolicy, s::TutoringState)
    """
    Rollout policy for POMCPOW value estimation
    Uses transformer to get template distribution, then samples action
    """

    # Get transformer prediction (uses caching internally via get_or_compute_forward_pass)
    speaker, delta, template_dist = get_or_compute_forward_pass(policy.pomdp, s)

    # Deterministic turn-taking: student provides problem (turn 0), then tutor responds, then alternates
    # Total turns = real turns + simulated turns
    # Turn 1 (odd): TUTOR responds to problem
    # Turn 2 (even): STUDENT responds
    # Turn 3 (odd): TUTOR responds, etc.
    total_turns = s.current_turn_count + length(s.simulated_template_history)

    template_dist_masked = copy(template_dist)

    if total_turns % 2 == 1
        # Odd turn: TUTOR speaks
        template_dist_masked[1:NUM_STUDENT_TEMPLATES] .= Float32(-1e9)  # Mask student templates
    else
        # Even turn: STUDENT speaks (responding to previous tutor action)
        template_dist_masked[(NUM_STUDENT_TEMPLATES+1):end] .= Float32(-1e9)  # Mask tutor templates

        # Filter student templates based on PREVIOUS tutor action
        if s.last_action !== nothing
            valid_templates = get(ACTION_STUDENT_TEMPLATE_FILTERS,
                                 s.last_action,
                                 1:NUM_STUDENT_TEMPLATES)

            # Create action filter mask: zero out invalid templates
            action_mask = zeros(Float32, length(template_dist_masked))
            for idx in valid_templates
                if idx <= NUM_STUDENT_TEMPLATES
                    action_mask[idx] = 1.0f0
                end
            end

            # Apply action filter (multiplicative - keeps speaker masking)
            template_dist_masked = template_dist_masked .+ (1.0f0 .- action_mask) .* Float32(-1e9)
        end
    end

    # Sample template
    template_probs = softmax(template_dist_masked)
    template_idx = sample(Random.default_rng(), 1:length(template_probs), Weights(template_probs))

    # Determine action based on speaker
    if template_idx <= NUM_STUDENT_TEMPLATES
        # Student template: action is always "listen" (tutor is listening to student)
        return listen
    else
        # Tutor template: sample pedagogical action from template's distribution
        return select_action_from_template(template_idx, Random.default_rng())
    end
end

# ========================
# POMCPOW SOLVER CONFIGURATION
# ========================

function create_pomcpow_solver(
    pomdp::CalculusTutoringPOMDP;
    n_iterations::Int=200,
    exploration_constant::Float32=20.0f0,
    tree_depth::Int=5,
    k_observation::Float64=5.0,
    alpha_observation::Float64=0.5
)
    """
    Create POMCPOW solver with specified parameters
    
    Parameters:
    - n_iterations: Number of simulations per action (tree_queries)
    - exploration_constant: UCB exploration parameter
    - tree_depth: Maximum planning depth
    - k_observation: Observation widening parameter
    - alpha_observation: Observation widening exponent
    """
    
    # Create rollout policy for value estimation
    rollout_policy = TutoringRolloutPolicy(pomdp)
    
    solver = POMCPOWSolver(
        tree_queries = n_iterations,
        criterion = MaxUCB(exploration_constant),
        max_depth = tree_depth,
        enable_action_pw = false,  # Don't use action progressive widening
        k_observation = k_observation,
        alpha_observation = alpha_observation,
        estimate_value = FORollout(rollout_policy),  # Use rollout for value estimation
        check_repeat_obs = true,
        rng = MersenneTwister(42)
    )
    
    return solver
end

# ========================
# Helper Functions
# ========================

function execute_action_in_environment(
    pomdp::CalculusTutoringPOMDP,
    s::TutoringState,
    a::TutoringAction
)
    """Execute action in real environment"""
    rng = Random.default_rng()
    result = POMDPs.gen(pomdp, s, a, rng)
    
    # Extract from NamedTuple
    return result.sp, result.o, result.r
end

function POMDPs.gen(pomdp::CalculusTutoringPOMDP, s::TutoringState, a::TutoringAction, rng::AbstractRNG)
    """
    Generative model: sample next state, observation, and reward
    """
    # Sample next state
    sp = rand(rng, transition(pomdp, s, a))
    
    # Get observation
    o = rand(rng, observation(pomdp, s, a, sp))
    
    # Calculate reward
    r = reward(pomdp, s, a, sp)
    
    return (sp=sp, o=o, r=r)
end