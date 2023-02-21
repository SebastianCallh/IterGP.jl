function trace_approx(A, zs)
    n = size(A, 1)
    ℓ = size(zs, 2)
    mapreduce(z -> n/ℓ * z'A*z, +, eachcol(zs))
end

function log_det_approx(K, P, zs)
    logdet(P) + trace_approx(log(K) - log(P), zs)
end