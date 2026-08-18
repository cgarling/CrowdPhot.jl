# Extracted and adapted from Bessels.jl (MIT-licensed, Michael Helton et al.).
# These functions have been modified to support SIMD through LoopVectorization.jl.
# This mainly involves changing `if else end` branches into `ifelse` statements,
# and dispatching on `eltype` rather than the concrete argument type so that
# `VectorizationBase.Vec{N,Float32}`/`Vec{N,Float64}` route to the same
# precision-specific body as their scalar element type.

# Polynomials
#
# Stored "structure-of-arrays": one plain `Vector{Float64}` per coefficient
# slot (16 intervals each), not a single tuple-of-tuples or matrix. This is
# what makes the table lookup in `_besselj1(x, ::Type{Float64})` actually
# vectorize under `@turbo`, not just avoid crashing:
#
# `VectorizationBase` defines `getindex(A::Array, i::AbstractSIMD,
# j::Vararg{AbstractSIMD,K})` (a real, ordinary-dispatch method, not
# something `@turbo`'s macro rewrites) that lowers to a genuine hardware
# gather when every index argument is a `Vec`. This method is declared for
# `Array` specifically -- not `AbstractArray` -- because gathering requires
# a real, strided memory pointer (`stridedpointer`) to compute per-lane
# offsets from; `Tuple` has no such pointer at all (indexing one with a
# `Vec` falls into Base's deprecated, non-`AbstractVector` fallback and
# recurses indefinitely: `StackOverflowError`, program state possibly
# corrupted) and `StaticArrays.SMatrix`/`SVector` don't either (they may
# live entirely in registers).
#
# It follows that `J1_POLYS_F64` itself -- the outer container -- must
# never be the thing indexed by `n`: `J1_POLYS_F64[n]` for a `Vec` `n` hits
# exactly the `Tuple`-indexing crash above (it did, here, in an earlier
# revision of this file). Each of the 14 coefficient slots must be gathered
# independently, `J1_POLYS_F64[k][n]` for literal `k` and `Vec` `n`, which
# also happens to be the only layout where SIMD lanes selecting *different*
# intervals even makes sense: `evalpoly(r, coeffs)` needs one `Vec` per
# coefficient, each lane's entry drawn from that lane's own interval, not
# one lane's whole 14-coefficient row.
#
# Built by transposing the 16 (one per pi/2-wide interval `n`) x 14 (one per
# coefficient) table upstream Bessels.jl stores row-major -- kept in that
# original layout here too so it stays easy to diff against upstream.
const J1_POLYS_F64 = ntuple(k -> Float64[row[k] for row in (
    (0.5818652242815964, 8.973834293036876e-17, -0.20511071214777496, 0.006058948324597746, 0.013801769808047723, -0.0003723170971916689, -0.0003949590750416506, 9.202953658715798e-6, 6.267292697927347e-6, -1.2681571675674473e-7, -6.319326736018798e-8, 1.057852721352679e-9, 4.839653511789343e-10, -2.1534683937074676e-11),
    (0.0, -0.402759395702553, 0.05255614585697725, 0.053410444132722765, -0.005179719245639115, -0.00223312533910116, 0.00017466429071755533, 4.6208700909637e-5, -3.0368632747366735e-6, -5.727790851165856e-7, 3.248220586398651e-8, 4.732693354840469e-9, -2.346792988667587e-10, -2.607612190048509e-11),
    (-0.34612620185379156, 2.6002760076991505e-17, 0.16697453550109428, -0.009678268542879972, -0.012099225779175138, 0.0006654009006841076, 0.0003541389011182841, -1.742720356958137e-5, -5.655294960447268e-6, 2.484316221112215e-7, 5.7101369594782676e-8, -2.2587104440771904e-9, -3.9854619701676605e-10, 1.854508641982197e-11),
    (0.0, 0.30011575252613254, -0.02138921280934146, -0.04697047894974007, 0.0031302917260424344, 0.0021055871432169443, -0.00012550790947431806, -4.499147505171648e-5, 2.401580334889493e-6, 5.665260872870135e-7, -2.7272058756924113e-8, -4.718921752439826e-9, 2.045523137027535e-10, 2.6923755198436e-11),
    (0.27329994163319987, -1.3511933704624036e-16, -0.13477468037992396, 0.005116340346495344, 0.010631861751990702, -0.000448743683849691, -0.0003268000185531749, 1.3382556752353113e-5, 5.363177082527157e-6, -2.0647461181851255e-7, -5.499892476485602e-8, 1.9780335410646812e-9, 3.8456897851545063e-10, -1.5543946040865833e-11),
    (0.0, -0.2497048770578432, 0.012272357555101388, 0.04041116939079181, -0.0019268187972545589, -0.0019115826893618788, 8.661729445925766e-5, 4.241116264477775e-5, -1.800978985430059e-6, -5.47159694982648e-7, 2.1682608288127194e-8, 4.629875997238198e-9, -1.6946465290961774e-10, -2.7001263147077087e-11),
    (-0.23330441717143408, 1.3347222236161232e-16, 0.11580092244607755, -0.0032489977328297747, -0.009372527206041913, 0.0003036138212749204, 0.00029804555521007126, -9.813819271763006e-6, -5.0242293237237345e-6, 1.6136482548346424e-7, 5.251755806094281e-8, -1.6213723779041528e-9, -3.7130918416720704e-10, 1.2840797195120732e-11),
    (0.0, 0.21835940724787295, -0.008194403183877394, -0.03577820957502993, 0.001319573612804675, 0.0017308725061602815, -6.200735153951766e-5, -3.938703730297954e-5, 1.3569938030086903e-6, 5.190652330962483e-7, -1.7082573736581586e-8, -4.464580500726025e-9, 1.3835552663370753e-10, 2.6508049073606443e-11),
    (0.20701265272531905, -1.1572365385200866e-16, -0.10303781563402706, 0.002289729546293356, 0.008432435058846974, -0.00021965941971569214, -0.000272721543802475, 7.36909520166486e-6, 4.682127415369e-6, -1.2611406291992532e-7, -4.9733737497256667e-8, 1.313333637862963e-9, 3.5584811981981077e-10, -1.0581432980865016e-11),
    (0.0, -0.19646537146865717, 0.005964112206447895, 0.032382122684890324, -0.0009720337562254532, -0.0015842303417456071, 4.6667442166602986e-5, 3.6572573450276185e-5, -1.0499611641790313e-6, -4.891787571238003e-7, 1.3609422867092438e-8, 4.2640592578446675e-9, -1.1321195999108497e-10, -2.5685037344174917e-11),
    (-0.18801748852581776, 9.806595982257605e-17, 0.09371909377243136, -0.0017233243363111925, -0.007714266760106843, 0.00016755862384546863, 0.000251827174474777, -5.7335177376628445e-6, -4.372425030172481e-6, 1.0045839852031556e-7, 4.698184959516747e-8, -1.0717704296017814e-9, -3.3952344362582674e-10, 8.773851724781324e-12),
    (0.0, 0.18006337534431555, -0.004589739858905968, -0.029776581472313136, 0.0007530284838491313, 0.0014660390526475475, -3.658900932751018e-5, -3.4137236390211275e-5, 8.366325091910564e-7, 4.6108475352995054e-7, -1.1047200388967438e-8, -4.058793425595929e-9, 9.363513647344476e-11, 2.4720051360744732e-11),
    (0.1734590492857464, -8.319859584484218e-17, -0.08653590193876039, 0.0013568189856187354, 0.007147216520714834, -0.00013295776598760658, -0.00023462959289705953, 4.6031437800947915e-6, 4.103196271673978e-6, -8.18373301294018e-8, -4.44403211204062e-8, 8.872617322304839e-10, 3.2359730989751983e-10, -7.354373969368619e-12),
    (0.0, -0.16718460047381806, 0.0036727588017285762, 0.02770273166133465, -0.0006050364924617241, -0.0013693112504440895, 2.9615596890234806e-5, 3.206096184277923e-5, -6.841058767433935e-7, -4.358877331737043e-7, 9.143763763164119e-9, 3.864037532317281e-9, -7.851667555762691e-11, -2.3728757802646805e-11),
    (-0.1618382095526585, 7.115722920799028e-17, 0.08078219522574913, -0.0011038527651152986, -0.006686444726790514, 0.00010870535083026479, 0.00022030197347228358, -3.791805814546456e-6, -3.871147931227952e-6, 6.80607764766372e-8, 4.2160129307780366e-8, -7.460427055677985e-10, -3.0872139357605097e-10, 6.24158293622982e-12),
    (0.0, 0.15672498625285222, -0.0030251499811055965, -0.026004046442225856, 0.0004996832448037306, 0.001288697907658241, -2.457760943647847e-5, -3.02856199504406e-5, 5.715983781675481e-7, 4.136275513873783e-7, -7.703846567878601e-9, -3.6853569596797437e-9, 6.676439988396374e-11, 2.277101500922319e-11),
)], 14)

const J1_POLY_PIO2_F64 = (0.5, -0.0624999999999989, 0.002604166666657291, -5.42534721917933e-5, 6.781683542660179e-7, -5.651361336587487e-9, 3.36191211106159e-11, -1.4511302591871352e-13)

# Interval roots paired with J1_POLYS_F64: J1_ROOTS_HI_F64[n]/J1_ROOTS_LO_F64[n]
# are the (hi, lo) root pair for the n-th pi/2-wide interval, used to form
# the reduced argument `r = x - root_hi - root_lo` that J1_POLYS_F64[k][n]
# is expanded about. Two plain `Vector`s, same reason as J1_POLYS_F64 above.
const J1_ROOTS_HI_F64 = [
    1.8411837813406593, 3.8317059702075125, 5.3314427735250325, 7.015586669815619,
    8.536316366346286, 10.173468135062722, 11.706004902592063, 13.323691936314223,
    14.863588633909034, 16.470630050877634, 18.015527862681804, 19.615858510468243,
    21.16436985918879, 22.760084380592772, 24.311326857210776, 25.903672087618382,
]
const J1_ROOTS_LO_F64 = [
    4.7898393919093694e-18, -1.5269184090088067e-16, 1.5109105349471405e-16, -9.414165653410389e-17,
    -1.5433871213307537e-16, 4.482162274768888e-16, 7.121366942298246e-16, 2.600408064718813e-16,
    -6.265788988781879e-16, -1.619019544798128e-15, -1.1196999448424267e-16, -1.004445634526616e-15,
    1.7024131380423588e-15, -4.925749373614922e-16, -2.614798558537172e-16, 4.894530726419825e-16,
]

# J0 tables, same structure-of-arrays layout and same Array (not Tuple)
# requirement as the J1 tables above: each of the 14 coefficient slots must
# be gathered independently, `J0_POLYS_F64[k][n]` for literal `k` and `Vec`
# `n`; see the J1_POLYS_F64 comment for the full SIMD rationale. Built by
# transposing the 16 (one per pi/2-wide interval `n`) x 14 (one per
# coefficient) table upstream Bessels.jl stores row-major.
const J0_POLYS_F64 = ntuple(k -> Float64[row[k] for row in (
    (0.0, -0.5191474972894669, 0.10793870175491979, 0.05660177443794914, -0.008657669593292222, -0.0021942003590739974, 0.0002643770365942964, 4.37291931443113e-5, -4.338825419759815e-6, -5.304927679598406e-7, 4.469819175606667e-8, 4.3284827345621115e-9, -3.135111000732148e-10, -2.628876489517534e-11),
    (-0.402759395702553, 2.476919088072758e-16, 0.20137969785127532, -0.017518715285670765, -0.013352611033152278, 0.0010359438492839443, 0.00037218755624680334, -2.4952042421113972e-5, -5.776086353844158e-6, 3.374317129436161e-7, 5.727482259215452e-8, -2.9561880489355444e-9, -3.905845672635605e-10, 1.971332566705736e-11),
    (0.0, 0.34026480655836816, -0.030820651425593214, -0.05298855286760721, 0.004631042145890388, 0.002257440229081131, -0.00017518572879518415, -4.6521091062878115e-5, 3.199785909661533e-6, 5.716500268232186e-7, -3.5112898510841466e-8, -4.684643389757727e-9, 2.562685034682206e-10, 2.7958958795750104e-11),
    (0.30011575252613254, -1.6640272822046001e-16, -0.15005787626306408, 0.007129737603121546, 0.011742619737383848, -0.0006260583453094324, -0.00035093119008693753, 1.7929701912295164e-5, 5.6239324892485754e-6, -2.668437501970219e-7, -5.6648488273749086e-8, 2.48117399780498e-9, 3.8876537586241154e-10, -1.6657136713437192e-11),
    (0.0, -0.27145229992838193, 0.015684124960953488, 0.044033774963413, -0.0025093022271948434, -0.0020603351551475315, 0.00011243486771159352, 4.482303558813413e-5, -2.288390108003442e-6, -5.679383588459768e-7, 2.693939234375692e-8, 4.737285529934781e-9, -2.0612709555352797e-10, -2.8163166483726606e-11),
    (-0.2497048770578432, 1.1807897766765572e-16, 0.12485243852891914, -0.0040907858517059345, -0.010102792347641438, 0.00038536375952213334, 0.00031859711440332953, -1.2373899600646271e-5, -5.3013932979548665e-6, 2.001098153528186e-7, 5.4711629662471434e-8, -1.9724572531751518e-9, -3.8121398193699247e-10, 1.3667679743782715e-11),
    (0.0, 0.23245983136472478, -0.009857064513825458, -0.03818600911162367, 0.0016073972920762946, 0.0018420433388794816, -7.581358465623415e-5, -4.159284549011371e-5, 1.650645590334915e-6, 5.425453494592871e-7, -2.0556207467977526e-8, -4.620018928884712e-9, 1.642028058414746e-10, 2.7701605444102412e-11),
    (0.21835940724787295, -8.89726402965429e-17, -0.10917970362393398, 0.0027314677279632535, 0.008944552393700088, -0.00026391472261453583, -0.00028847875053074687, 8.858193371737123e-6, 4.9233776180403375e-6, -1.5077786827161215e-7, -5.190218733666561e-8, 1.5539413886301204e-9, 3.674809363354973e-10, -1.1113645791216594e-11),
    (0.0, -0.20654643307799603, 0.006916736034268416, 0.034115572697347704, -0.001137276252948717, -0.0016680057255530109, 5.4841792064182565e-5, 3.837965853474541e-5, -1.2335804050962046e-6, -5.106259295634553e-7, 1.592333632709497e-8, 4.423517565793139e-9, -1.3138837384184105e-10, -2.6809397212536384e-11),
    (-0.19646537146865717, 6.979167865106427e-17, 0.09823268573432613, -0.001988037402152532, -0.008095530671166083, 0.00019440675128712672, 0.0002640383898036336, -6.666777683303928e-6, -4.5715696772304925e-6, 1.1666296153560847e-7, 4.8913639696764225e-8, -1.2379867207945651e-9, -3.508930968415813e-10, 9.07632091591013e-12),
    (0.0, 0.18772880304043943, -0.005194182350684612, -0.031096513233785917, 0.0008577442641273341, 0.0015312251534677639, -4.184307585284775e-5, -3.5603170534217916e-5, 9.58002109234601e-7, 4.795250964600283e-7, -1.263434500625308e-8, -4.20550167685701e-9, 1.065791122326672e-10, 2.5727417675684577e-11),
    (0.18006337534431555, -5.638484737644332e-17, -0.09003168767215539, 0.0015299132863046024, 0.0074441453680234426, -0.00015060569680378768, -0.0002443398416353605, 5.227001519013193e-6, 4.267152607972633e-6, -9.295966007495808e-8, -4.610438011417262e-8, 1.0049092632275165e-9, 3.339442682325105e-10, -7.4991079301099e-12),
    (0.0, -0.17326589422922986, 0.004084217951979124, 0.028749284970146657, -0.0006761643016121907, -0.0014215899173758441, 3.320978125391802e-5, 3.3264379323815026e-5, -7.684444100376941e-7, -4.515479818833484e-7, 1.0271599456634145e-8, 3.993903280527723e-9, -8.793975528824604e-11, -2.4610374225652004e-11),
    (-0.16718460047381803, 4.662597138876655e-17, 0.08359230023690671, -0.0012242529339116864, -0.006925682915280748, 0.00012100729852042988, 0.00022821854128498174, -4.230799709959437e-6, -4.007618360337058e-6, 7.601217108010105e-8, 4.358483157250522e-8, -8.317621452517008e-10, -3.178805429572483e-10, 6.284538399593043e-12),
    (0.0, 0.16170155068925002, -0.0033200234006036146, -0.02685937038656159, 0.0005505380905909195, 0.0013316994659124243, -2.715683205388875e-5, -3.1289544794362e-5, 6.326900360777323e-7, 4.2697141781883964e-7, -8.532447325590315e-9, -3.799009445641295e-9, 7.379552811590934e-11, 2.353654960834717e-11),
    (0.15672498625285222, 1.1464880342445208e-19, -0.07836249312642612, 0.0010083833270351796, 0.006501011610557426, -9.993664895655577e-5, -0.00021478298462967253, 3.511086890959515e-6, 3.7857022791122945e-6, -6.350944888510364e-8, -4.135588012575766e-8, 7.135988717747828e-10, 3.2075281131621564e-10, 2.208506585533542e-12),
)], 14)

const J0_POLY_PIO2_F64 = (1.0, -0.25, 0.01562499999999994, -0.00043402777777725544, 6.781684026082576e-6,
    -6.781683757550061e-8, 4.709479394601058e-10, -2.4016837144506874e-12, 9.104258208703104e-15)

# Interval roots paired with J0_POLYS_F64: J0_ROOTS_HI_F64[n]/J0_ROOTS_LO_F64[n]
# are the (hi, lo) root pair for the n-th pi/2-wide interval, used to form
# the reduced argument `r = x - root_hi - root_lo` that J0_POLYS_F64[k][n]
# is expanded about. Two plain `Vector`s, same reason as J0_POLYS_F64 above.
const J0_ROOTS_HI_F64 = [
    2.4048255576957730, 3.8317059702075125, 5.5200781102863110, 7.0155866698156190,
    8.6537279129110130, 10.173468135062722, 11.791534439014281, 13.323691936314223,
    14.930917708487787, 16.470630050877634, 18.071063967910924, 19.615858510468243,
    21.211636629879260, 22.760084380592772, 24.352471530749302, 25.903672087618382,
]
const J0_ROOTS_LO_F64 = [
    -1.176691651530894e-16, -1.5269184090088067e-16, 8.088597146146722e-17, -9.414165653410389e-17,
    -2.92812607320779e-16, 4.482162274768888e-16, 2.812956912778735e-16, 2.600408064718813e-16,
    -7.070514505983074e-16, -1.619019544798128e-15, -9.658048089426209e-16, -1.004445634526616e-15,
    4.947077428784068e-16, -4.925749373614922e-16, 9.169067133951066e-16, 4.894530726419825e-16,
]

# Math constants
const THPIO4(::Type{Float64}) = 2.35619449019234492885
const THPIO4(::Type{Float32}) = 2.35619449019234492885f0
const PIO4(::Type{Float64}) = 0.78539816339744830962
const PIO4(::Type{Float32}) = 0.78539816339744830962f0
const SQ2OPI(::Type{Float64}) = 0.79788456080286535588
const TWOOPI(::Type{Float64}) = 0.6366197723675814

# Other constants
const JP32 = (
    -3.405537384615824f-2, 1.937383947804541f-3, -4.541343896997497f-5,
    6.009061827883699f-7, -4.878788132172128f-9
)

const MO132 = (
    7.978845453073848f-1, 4.976029650847191f-6, 1.493389585089498f-1,
    5.435364690523026f-3, -2.102302420403875f-1, 3.138238455499697f-1,
    -2.284801500053359f-1,6.913942741265801f-2,
)

const PH132 = (
    3.749989509080821f-1, -1.637986776941202f-1, 3.503787691653334f-1,
    -1.544842782180211f0, 7.222973196770240f0, -2.485774108720340f1,
    5.073465654089319f1, -4.497014141919556f1,
)

const JP032 = (
    -1.729150680240724f-1, 1.332913422519003f-2, -3.969646342510940f-4,
    6.388945720783375f-6, -6.068350350393235f-8
)

const MO032 = (
    7.978845717621440f-1, -3.355424622293709f-6, -4.969382655296620f-2,
    -3.560281861530129f-3, 1.197549369473540f-1, -2.145007480346739f-1,
    1.864949361379502f-1, -6.838999669318810f-2
)

const PH032 = (
    -1.249992184872738f-1, 6.490598792654666f-2, -1.939906941791308f-1,
    1.001973420681837f0, -4.974978466280903f0, 1.756221482109099f1,
    -3.630592630518434f1, 3.242077816988247f1,
)

# Implementations

"""
    besselj0(x::T) where T <: Union{Float32, Float64, ComplexF32, ComplexF64}

Bessel function of the first kind of order zero, ``J_0(x)``.
"""
besselj0(x::Real) = _besselj0(float(x))

_besselj0(x) = _besselj0(x, eltype(x))

_besselj0(x, ::Type{Float16}) = Float16(_besselj0(Float32(x), Float32))

# Branchless port of Bessels.jl's `_besselj0(x::Float64)` (3 regions: a
# 9-term even minimax polynomial for x <= pi/2, a per-interval polynomial
# selected by table lookup for pi/2 < x < 26, and an amplitude*sin(phase)
# asymptotic form for x >= 26, itself split into two rational-approximation
# orders at x = 120). All three regions are evaluated unconditionally to
# support SIMD vectorization.
#
# The same two simplifications relative to upstream Bessels.jl as in
# `_besselj1(x, ::Type{Float64})`, both irrelevant to this package's actual
# domain (u = pi*r/a for a pixel-scale radius r and PSF-scale a):
#   - the table-lookup interval index `n` is clamped into the valid table
#     range (rather than implicitly guaranteed in-range by an enclosing
#     branch), so the lookup never goes out of bounds even when this region
#     isn't the one actually selected by the final `ifelse`;
#   - the asymptotic region uses plain `sin` instead of upstream's `sin_sum`
#     (upstream computes sin(x + pi/4 + xn) with a double-double argument
#     reduction built on private `Base.Math` internals). Verified against
#     `Bessels.besselj0` (see test/bessels.jl) with the same accuracy
#     envelope as `_besselj1`: exact to ~1e-15 absolute / ~1e-10 relative
#     for x up to 1e6, degrading only to ~1e-6 relative by x ~ 1e12.
# `sin`'s argument is separately clamped finite (`phase_safe`) for the same
# x==Inf reason documented on `_besselj1(x, ::Type{Float64})` (here the
# `0 * sin(Inf) -> NaN` poison would arrive through `a`, which already
# correctly carries the x==Inf -> 0 result through `xinv = inv(x_safe) == 0`).
@inline function _besselj0(x, ::Type{Float64})
    T = Float64
    x = abs(x)

    val_pio2 = evalpoly(x * x, J0_POLY_PIO2_F64)

    n = clamp(unsafe_trunc(Int, TWOOPI(T) * x), 1, 16)
    root_hi = @inbounds J0_ROOTS_HI_F64[n]
    root_lo = @inbounds J0_ROOTS_LO_F64[n]
    r = x - root_hi - root_lo
    P = J0_POLYS_F64
    coeffs = @inbounds (P[1][n], P[2][n], P[3][n], P[4][n], P[5][n], P[6][n], P[7][n],
        P[8][n], P[9][n], P[10][n], P[11][n], P[12][n], P[13][n], P[14][n])
    val_mid = evalpoly(r, coeffs)

    x_safe = ifelse(x < 26.0, 27.0, x) # keep inv/sqrt away from the unused branch's singular region
    xinv = inv(x_safe)
    x2 = xinv * xinv
    p1 = (1.0, -1 / 16, 53 / 512, -4447 / 8192, 3066403 / 524288, -896631415 / 8388608, 796754802993 / 268435456, -500528959023471 / 4294967296)
    q1 = (-1 / 8, 25 / 384, -1073 / 5120, 375733 / 229376, -55384775 / 2359296, 24713030909 / 46137344, -7780757249041 / 436207616)
    p2 = (1.0, -1 / 16, 53 / 512, -4447 / 8192)
    q2 = (-1 / 8, 25 / 384, -1073 / 5120, 375733 / 229376)
    use_far = x_safe >= 120.0
    p = ifelse(use_far, evalpoly(x2, p2), evalpoly(x2, p1))
    q = ifelse(use_far, evalpoly(x2, q2), evalpoly(x2, q1))
    a = SQ2OPI(T) * sqrt(xinv) * p
    phase = x_safe + muladd(xinv, q, PIO4(T))
    phase_safe = ifelse(isfinite(phase), phase, zero(T))
    val_asym = a * sin(phase_safe)

    val = ifelse(x <= pi / 2, val_pio2, ifelse(x < 26.0, val_mid, val_asym))
    return val
end
@inline function _besselj0(x, ::Type{Float32})
    T = Float32
    x = abs(x)
    small = x <= 2.0f0

    z = x * x
    DR1 = 5.78318596294678452118f0
    p_small = ifelse(x < 1.0f-3, 1.0f0 - 0.25f0 * z, (z - DR1) * evalpoly(z, JP032))

    x_safe = ifelse(small, 3.0f0, x) # keep inv/sqrt away from the unused branch's singular region
    q = inv(x_safe)
    w = sqrt(q)
    p_amp = w * evalpoly(q, MO032)
    w = q * q
    xn = q * evalpoly(w, PH032) - PIO4(T)
    phase = xn + x_safe
    phase_safe = ifelse(isfinite(phase), phase, zero(T))
    p_large = p_amp * cos(phase_safe)

    return ifelse(small, p_small, p_large)
end

"""
    besselj1(x::T) where T <: Union{Float32, Float64, ComplexF32, ComplexF64}

Bessel function of the first kind of order one, ``J_1(x)``.
"""
besselj1(x::Real) = _besselj1(float(x))

_besselj1(x) = _besselj1(x, eltype(x))

_besselj1(x, ::Type{Float16}) = Float16(_besselj1(Float32(x), Float32))

# Branchless port of Bessels.jl's `_besselj1(x::Float64)` (3 regions: a
# low-order polynomial for x <= pi/2, a per-interval polynomial selected by
# table lookup for pi/2 < x <= 26, and an amplitude*cos(phase) asymptotic
# form for x > 26, itself split into two rational-approximation orders at
# x = 120). All three regions are evaluated unconditionally and blended with
# `ifelse`, same approach as `_besselj1(x, ::Type{Float32})` above.
#
# Two simplifications relative to upstream Bessels.jl, both irrelevant to
# this package's actual domain (u = pi*r/a for a pixel-scale radius r and
# PSF-scale a -- never remotely close to these boundaries):
#   - the table-lookup interval index `n` is clamped into the valid table
#     range (rather than implicitly guaranteed in-range by an enclosing
#     branch), so the lookup never goes out of bounds even when this region
#     isn't the one actually selected by the final `ifelse`;
#   - the asymptotic region uses plain `cos` instead of upstream's
#     `cos_sum`, which uses a double-double argument reduction built on
#     private `Base.Math` internals to stay accurate as x grows into the
#     range where `x + xn` itself starts losing bits of `xn`. Verified
#     against `Bessels.besselj1` (see test/bessels.jl): exact to ~1e-15
#     absolute / ~1e-10 relative for x up to 1e6, degrading only to ~1e-6
#     relative by x ~ 1e12.
# `cos`'s argument is separately clamped finite (`phase_safe`): for x==Inf,
# `x_safe` is *not* substituted away (x > 26 is the branch actually
# selected), so `cos(x_safe + xn)` would otherwise evaluate `cos(Inf)` --
# Base's scalar `cos` throws `DomainError` there, and even where it doesn't
# throw (`Vec` lanes under `@turbo`), `cos(Inf) -> NaN` would poison the
# `a * cos(...)` product via `0 * NaN -> NaN` despite `a` already correctly
# carrying the x==Inf -> 0 result through `xinv = inv(x_safe) == 0`.
@inline function _besselj1(x, ::Type{Float64})
    T = Float64
    s = sign(x)
    x = abs(x)

    val_pio2 = x * evalpoly(x * x, J1_POLY_PIO2_F64)

    n = clamp(unsafe_trunc(Int, TWOOPI(T) * x), 1, 16)
    root_hi = @inbounds J1_ROOTS_HI_F64[n]
    root_lo = @inbounds J1_ROOTS_LO_F64[n]
    r = x - root_hi - root_lo
    P = J1_POLYS_F64
    coeffs = @inbounds (P[1][n], P[2][n], P[3][n], P[4][n], P[5][n], P[6][n], P[7][n],
        P[8][n], P[9][n], P[10][n], P[11][n], P[12][n], P[13][n], P[14][n])
    val_mid = evalpoly(r, coeffs)

    x_safe = ifelse(x <= 26.0, 27.0, x) # keep inv/sqrt away from the unused branch's singular region
    xinv = inv(x_safe)
    x2 = xinv * xinv
    p1 = (1.0, 3 / 16, -99 / 512, 6597 / 8192, -4057965 / 524288, 1113686901 / 8388608, -951148335159 / 268435456, 581513783771781 / 4294967296)
    q1 = (3 / 8, -21 / 128, 1899 / 5120, -543483 / 229376, 8027901 / 262144, -30413055339 / 46137344, 9228545313147 / 436207616)
    p2 = (1.0, 3 / 16, -99 / 512, 6597 / 8192)
    q2 = (3 / 8, -21 / 128, 1899 / 5120, -543483 / 229376)
    use_far = x_safe >= 120.0
    p = ifelse(use_far, evalpoly(x2, p2), evalpoly(x2, p1))
    q = ifelse(use_far, evalpoly(x2, q2), evalpoly(x2, q1))
    a = SQ2OPI(T) * sqrt(xinv) * p
    xn = muladd(xinv, q, -3 * PIO4(T))
    phase = x_safe + xn
    phase_safe = ifelse(isfinite(phase), phase, zero(T))
    val_asym = a * cos(phase_safe)

    val = ifelse(x <= pi / 2, val_pio2, ifelse(x <= 26.0, val_mid, val_asym))
    return val * s
end
@inline function _besselj1(x, ::Type{Float32})
    T = Float32
    s = sign(x)
    x = abs(x)
    small = x <= 2.0f0

    z = x * x
    Z1 = 1.46819706421238932572f1
    p_small = (z - Z1) * x * evalpoly(z, JP32)

    x_safe = ifelse(small, 3.0f0, x) # keep inv/sqrt away from the unused branch's singular region
    q = inv(x_safe)
    w = sqrt(q)
    p_amp = w * evalpoly(q, MO132)
    w = q * q
    xn = q * evalpoly(w, PH132) - THPIO4(T)
    phase = xn + x_safe
    phase_safe = ifelse(isfinite(phase), phase, zero(T)) # x==Inf guard
    p_large = p_amp * cos(phase_safe)

    return ifelse(small, p_small, p_large) * s
end
