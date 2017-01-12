import base64
import struct


def start():

    feature_base1 = "El5xvkEpeD4KFY6+ewH6PvKbNz5lrZY+y1n2PmPgrr6CqFe9OMHgPjdPXD9l218//PnZvMRxnT6h+VG+mhCwPoGZDr6NnQ4+SDIjPzSPIr9UzeO+d6JNvpYQTz58gwq+g4qxPid/Ob/CESC+Eranv4jvnr+LM2q+oL/SPpYoP71Oim0/Jx6KP9jDpr8KaMs/3iI9vn+lEj8LlwU/B7KLvXR08T6ukHA/KZi0PsipjT5QtOy79rQlPS4PpT9jch0+bzFkvxpisT46KCo/OeQvP+XnYT8AP68+gCXyvkVu4762Bls9+7h3v+5mU78Pwa8/Ghssv8GMar6MkMK+MIy8PrpRij+R0pa+atOkv2K4w74H04A/q1WVPQDa9r3QWpg9uHMIP10L3D3QeMA/OAjxvx+LWD9cdlE/HzndvqJaMr8hY9W94QiiP6mDzj1ikF890UdlP0sqyD5DeAi+o5kXPiA/8D6KOR8+EGNLvxLtwT3BTS6/K1dNP9GTTj8p/aQ/CckNv0ZB3j6JyYG+USw7P54jw75icoA+dmTFvvpTkD+P/f09lDmVPmwwg74IT1Y+owSLvZSNBD7oKJM+d04yvyLBdD7CHSE/O2s5PxxvFL+xX+s+BzbKPjhmez5lpTK+ipUhP2QYNz8MhKi+hBL/PJpun756nko/9xTUPk5ye74="
    feature_base2 = "HizVviOYmj5CvJq+iHs5Pisv5L1+8t0+zABaP0yH7L4UTuc8ViXrPuvdRj8O+nU/GQqgPohkEj+G2Jm+aNn1PuEPNj4whcg+V3zLPnnQCL94gN++pOcCPvmdxj4Okj++tb+6Pqyn8L6HK5s+s+JmvyGZmr/A79i7qEVYPpEbXD5jcEY/opNiPwqxIb+jR8M/eDS8vn5BJT8gioA+guSRvgeBCj/RfIE/e3ihPs5/Kr04JTA8YZu8PfsUej8SPXw+0MR8v6i3AT85nt0+61sJP/dejD9ykRQ/PgPJvvjrh7x4xbO9BV98v4MGAr90TJY/dAtYvrrhyr29WTy+8kHBPkzWuj+3aKG+oXZPv5neor6rTok/x2nyvbT8pj3LRDY9AZsQP2J+zD28u7o/Bva9v5h6Lz+A0Sk/7OUKv27yB78Qi6i7QxB6P54nuj7gehm+Hp1jP50axj6W1+q9KycWPltJJD8dgf8+uyYBv1o8ub1ERCy/ctA9PkuUOT+tfps/uSyFvgq36z7x6su+FE83P6K73T16Uy8+XOWyvg20lT+4F428L89hPiS0r72ciHA+b1KGvgjGWj7CjdY9hfASv5KKMT52nEA/J+IuPyCZCL/ypME9gOC3PWDPYLyS9dk9YM5JPgS46j5p9p2+ISwOPzCyzr3cVIA/wt03PzArXT4="
    feature_s1 = base64.b64decode(feature_base1)
    feature_s2 = base64.b64decode(feature_base2)
    feature1 = struct.unpack("128f", feature_s1)
    feature2 = struct.unpack("128f", feature_s2)

    score = 0
    for i in range(0, len(feature1)):
        length = (feature1[i] - feature2[i])
        score = score + (length * length)

    r = -0.02 * score + 1.1;

    if r <= 0.0:
        r = 0.0
    elif r > 1.0:
        r = 1.0

    print r


if __name__ == "__main__":
    start()
