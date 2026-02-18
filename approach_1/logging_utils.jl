using Dates

if !isdefined(Main, :_extract_n_from_argv)
    function _extract_n_from_argv(args)
        for a in args
            startswith(a, "-") && continue
            try
                return parse(Int, a)
            catch
            end
        end
        return nothing
    end
end

if !isdefined(Main, :run_with_terminal_log)
    function run_with_terminal_log(f::Function, runner_name::String, args::Vector{String})
        n = _extract_n_from_argv(args)
        n_dir = isnothing(n) ? "unknown" : string(n)
        ts = Dates.format(now(), "yyyymmdd_HHMMSS")
        logs_dir = joinpath(@__DIR__, "logs", n_dir)
        mkpath(logs_dir)
        log_path = joinpath(logs_dir, "$(runner_name)-$(ts).log")

        orig_stdout = stdout
        orig_stderr = stderr
        rc = 1

        tee_cmd = pipeline(`tee $(log_path)`, stdout=orig_stdout, stderr=orig_stderr)
        open(tee_cmd, "w") do io
            rc = redirect_stdout(io) do
                redirect_stderr(io) do
                    try
                        return f()
                    catch e
                        showerror(stderr, e, catch_backtrace())
                        println(stderr)
                        return 1
                    end
                end
            end
        end

        println(orig_stdout, "Log saved: $(log_path)")
        return rc
    end
end
