"""
Simple HTTP Server for Julia (No SSL dependencies)
Uses Sockets instead of HTTP.jl to avoid OpenSSL initialization issues
"""

using Sockets
using JSON

mutable struct SimpleHTTPServer
    port::Int
    router::Function
end

function parse_http_request(client::TCPSocket)
    """Parse HTTP request from TCP socket"""
    # Read request line
    request_line = readline(client)
    parts = split(request_line, " ")
    if length(parts) < 3
        return nothing
    end

    method = parts[1]
    target = parts[2]

    # Read headers
    headers = Dict{String, String}()
    content_length = 0

    while true
        line = readline(client)
        if isempty(line) || line == "\r"
            break
        end

        # Parse header
        if occursin(":", line)
            key, value = split(line, ":", limit=2)
            headers[strip(lowercase(key))] = strip(value)

            if lowercase(key) == "content-length"
                content_length = parse(Int, strip(value))
            end
        end
    end

    # Read body if present
    body = ""
    if content_length > 0
        body = String(read(client, content_length))
    end

    return (method=method, target=target, headers=headers, body=body)
end

function send_http_response(client::TCPSocket, status::Int, body::String, content_type::String="application/json")
    """Send HTTP response to TCP socket"""
    status_text = status == 200 ? "OK" :
                  status == 404 ? "Not Found" :
                  status == 500 ? "Internal Server Error" :
                  "Unknown"

    response = "HTTP/1.1 $status $status_text\r\n"
    response *= "Content-Type: $content_type\r\n"
    response *= "Content-Length: $(length(body))\r\n"
    response *= "Connection: close\r\n"
    response *= "\r\n"
    response *= body

    write(client, response)
end

function serve(server::SimpleHTTPServer)
    """Start the simple HTTP server"""
    listener = Sockets.listen(server.port)
    println("Simple HTTP server listening on port $(server.port)")

    try
        while true
            client = accept(listener)

            try
                # Parse request
                request = parse_http_request(client)

                if request === nothing
                    send_http_response(client, 400, JSON.json(Dict("error" => "Bad Request")))
                    close(client)
                    continue
                end

                # Call router
                response = server.router(request)

                # Send response
                send_http_response(client, response.status, response.body, get(response.headers, "Content-Type", "application/json"))

            catch e
                # Write to file immediately to bypass pipe buffering
                try
                    error_log = open("/tmp/julia_errors.log", "a")
                    println(error_log, "\n" * "="^70)
                    println(error_log, "HTTP SERVER ERROR AT $(now())")
                    println(error_log, "="^70)
                    println(error_log, "\n❌ ERROR HANDLING REQUEST:")
                    println(error_log, e)
                    showerror(error_log, e, catch_backtrace())
                    println(error_log, "\n" * "="^70 * "\n")
                    close(error_log)
                catch file_err
                    println("Failed to write to error log: $file_err")
                end

                # Also write to stdout (though it may not be visible)
                println("\n❌ ERROR HANDLING REQUEST:")
                println(e)
                showerror(stdout, e, catch_backtrace())
                println()
                flush(stdout)  # CRITICAL: Flush error output immediately
                try
                    send_http_response(client, 500, JSON.json(Dict("error" => "Internal Server Error", "message" => string(e))))
                catch
                    # If we can't even send error response, just close
                end
            finally
                close(client)
            end
        end
    catch e
        if !isa(e, InterruptException)
            println("Server error: $e")
        end
    finally
        close(listener)
    end
end

# Response type
struct HTTPResponse
    status::Int
    body::String
    headers::Dict{String, String}

    HTTPResponse(status::Int, body::String) = new(status, body, Dict("Content-Type" => "application/json"))
    HTTPResponse(status::Int, body::String, content_type::String) = new(status, body, Dict("Content-Type" => content_type))
end
