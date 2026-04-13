using CSV
using DataFrames

function save_to_csv(Psave, Θsave, name)

    A = Psave'
    z = A[:, 3]

    ids = cumsum([0; z[2:end] .!= z[1:end-1]])

    A[:, 3] = ids

    data = hcat(A, Θsave)
    df = DataFrame(data, [:x, :y, :z, :val])
    println(df)
    CSV.write("data/insert_formatted/$name.csv", df, header= ["X (mm)", "Y(mm)", "RingNumber", "Angle (deg)"])

end