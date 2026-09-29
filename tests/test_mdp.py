import unittest
import warnings

from gmx_harness import AWH, EM, MD, BarMethod, MartiniEM, MartiniMD, MDParameters, MDPValidationError, MDType
from gmx_harness.mdp import V_RESCALE_C_RESCALE_MDP, validate_mdp_text


def errors(m: MDParameters) -> list[str]:
    return [i.key for i in m.validate() if i.level == "error"]


class TestMDParameters(unittest.TestCase):
    def test_parse_and_export(self) -> None:
        m = MDParameters.from_text("; comment\nnsteps = 10 ; trailing\n\ndt=0.002\n")
        # trailing comments are kept (mylibs behaviour) but ignored by validation
        self.assertEqual(m.data, {"nsteps": "10 ; trailing", "dt": "0.002"})
        self.assertEqual(m.export(), "nsteps = 10 ; trailing\ndt     = 0.002")
        self.assertEqual(m.validate(), [])

    def test_equivalent_spelling_is_replaced_in_place(self) -> None:
        m = MDParameters({"tc_grps": "system", "tau_t": "1"})
        m.add_or_update("tc-grps", "A B")
        self.assertEqual(list(m.data), ["tc-grps", "tau_t"])
        self.assertEqual(m.get("TC_GRPS"), "A B")

    def test_placeholder_left_over(self) -> None:
        self.assertIn("nsteps", errors(MDParameters(V_RESCALE_C_RESCALE_MDP)))

    def test_newline_injection(self) -> None:
        self.assertEqual(errors(MDParameters({"define": "-DX\nnsteps = 0"})), ["define"])

    def test_numbers_and_ranges(self) -> None:
        self.assertEqual(errors(MDParameters({"dt": "abc"})), ["dt"])
        self.assertEqual(errors(MDParameters({"dt": "0"})), ["dt"])
        self.assertEqual(errors(MDParameters({"nsteps": "-1"})), [])
        self.assertEqual(errors(MDParameters({"gen_vel": "maybe"})), ["gen_vel"])

    def test_group_counts(self) -> None:
        m = MDParameters({"tcoupl": "v-rescale", "tc_grps": "A B", "tau_t": "0.1", "ref_t": "300 300"})
        self.assertEqual(errors(m), ["tau_t"])
        m = MDParameters({"pcoupl": "c-rescale", "pcoupltype": "semiisotropic", "ref_p": "1", "compressibility": "4.5e-5 4.5e-5"})
        self.assertEqual(errors(m), ["ref_p"])

    def test_grompp_output_mdp_is_accepted(self) -> None:
        # grompp writes the seed it drew (often negative) and GROMACS-2026-only options into output.mdp
        text = "gen-seed = -1591849\nld-seed = -2633729\nnnpot-active = no\ncolvars-active = no\nQMMM = no\n"
        self.assertEqual(validate_mdp_text(text), [])
        self.assertEqual(errors(MDParameters({"gen_seed": "abc"})), ["gen_seed"])

    def test_duplicate_spelling(self) -> None:
        m = MDParameters({"Pcoupl": "no", "pcoupl": "no"})
        self.assertEqual(errors(m), ["pcoupl"])

    def test_unknown_key_is_only_a_warning(self) -> None:
        issues = validate_mdp_text("integrater = md\nawh1-dim1-start = 1\npull-ncoords = 1")
        self.assertEqual([(i.level, i.key) for i in issues], [("warning", "integrater")])

    def test_ensure_valid(self) -> None:
        with self.assertRaises(MDPValidationError):
            MDParameters({"dt": "-1"}).ensure_valid()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            MDParameters({"dt": "-1"}).ensure_valid(strict=False)
        self.assertTrue(w)

    def test_steps_validate_on_generate(self) -> None:
        md = MD(type=MDType.v_rescale_c_rescale, calculation_name="md",
                additional_mdp_parameters={"tc_grps": "A B"})
        with self.assertRaises(MDPValidationError):
            md.generate()
        # the escape hatch
        md2 = MD(type=MDType.v_rescale_c_rescale, calculation_name="md",
                 additional_mdp_parameters={"tc_grps": "A B"}, strict_mdp=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertIn("setting.mdp", md2.generate())

    def test_all_templates_generate_cleanly(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            EM().generate()
            for t in (MDType.v_rescale_c_rescale, MDType.v_rescale_only_nvt):
                MD(type=t, calculation_name="x").generate()
            MD(type=MDType.nose_hoover_parinello_rahman, calculation_name="x", gen_vel="no").generate()
            MD(type=MDType.v_rescale_c_rescale, calculation_name="x", useSemiisotropic=True).generate()
            MartiniEM().generate()
            MartiniMD(calculation_name="cg").generate()
            MartiniMD(calculation_name="cg", useSemiisotropic=True).generate()
            AWH().generate()
            zeros = [0.0, 0.0]
            BarMethod(type=MDType.v_rescale_c_rescale, calculation_name="bar", vdw_lambdas=[0.0, 1.0],
                      coul_lambdas=zeros, bonded_lambdas=zeros, restraint_lambdas=zeros, mass_lambdas=zeros,
                      temperature_lambdas=zeros).generate()

    def test_continuation_is_selectable(self) -> None:
        def cont(**kw: object) -> object:
            text = MD(type=MDType.nose_hoover_parinello_rahman, calculation_name="x", gen_vel="no", **kw).generate()
            return MDParameters.from_text(text["setting.mdp"]).get("continuation")

        self.assertEqual(cont(), "yes")  # template value (mylibs)
        self.assertEqual(cont(continuation=False), "no")
        self.assertEqual(cont(continuation=True), "yes")
        text = MD(type=MDType.v_rescale_c_rescale, calculation_name="x", continuation=True, gen_vel="no").generate()
        self.assertEqual(MDParameters.from_text(text["setting.mdp"]).get("continuation"), "yes")
        text = MD(type=MDType.v_rescale_c_rescale, calculation_name="x").generate()
        self.assertIsNone(MDParameters.from_text(text["setting.mdp"]).get("continuation"))

    def test_continuation_with_gen_vel_warns(self) -> None:
        issues = MDParameters({"continuation": "yes", "gen_vel": "yes"}).validate()
        self.assertEqual([(i.level, i.key) for i in issues], [("warning", "continuation")])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            MD(type=MDType.nose_hoover_parinello_rahman, calculation_name="x").generate()
        self.assertTrue(any("continuation=yes with gen_vel=yes" in str(x.message) for x in w))

    def test_md_fills_template_fields(self) -> None:
        # (field-by-field parity with mylibs was checked against mylibs itself when porting;
        # this pins the values so they cannot drift)
        mdp = MDParameters.from_text(
            MD(type=MDType.v_rescale_c_rescale, calculation_name="x", nsteps=123, nstout=7, temperature=310,
               defines=["POSRES"], useRestraint=True).generate()["setting.mdp"]
        )
        self.assertEqual(mdp.get("nsteps"), "123")
        self.assertEqual(mdp.get("nstxout"), "7")
        self.assertEqual(mdp.get("ref_t"), "310.0")  # float field, as in mylibs
        self.assertEqual(mdp.get("define"), "-DPOSRES")
        self.assertEqual(mdp.get("refcoord_scaling"), "all")


if __name__ == "__main__":
    unittest.main()
