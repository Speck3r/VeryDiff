@info "Building Sysimage..."
using PackageCompiler
deps_dir = @__DIR__
create_sysimage(
    ["VeryDiff"];
    sysimage_path="$deps_dir/VeryDiffAutoDiff.so",
    precompile_execution_file="$deps_dir/sysimage/trace_run_autodiff.jl",
    incremental=true
)